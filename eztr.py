import contextlib
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

CHECKPOINT_FORMAT = 2

# Architecture settings are always restored from a checkpoint.
ARCH_DEFAULTS = {
    "eed": 256,            # Encoder embedding dimension
    "ehs": 1024,           # Encoder hidden size (SwiGLU inner width)
    "enl": 4,              # Encoder number of layers
    "eah": 4,              # Encoder attention heads
    "ded": 256,            # Decoder embedding dimension
    "dhs": 1024,           # Decoder hidden size (SwiGLU inner width)
    "dnl": 4,              # Decoder number of layers
    "dah": 4,              # Decoder attention heads
    "use_rope": True,      # Rotary position embeddings (False: additive sinusoidal)
    "rope_theta": 10000.0, # RoPE base frequency
}

# Training/inference settings are restored from a checkpoint unless passed explicitly.
SETTINGS_DEFAULTS = {
    "drp": 0.3,                  # Dropout
    "bts": 800,                  # Batch size
    "lrt": 0.001,                # Peak learning rate
    "lst": 0.1,                  # Label smoothing
    "cnm": 1.0,                  # Clip norm
    "wup": 0.05,                 # Warmup, as a fraction of total training steps
    "optimizer": "adam",         # 'adam' or 'adamw'
    "adam_betas": (0.9, 0.999),
    "max_pred_len": None,        # None: max(50, 2 * source length + 10)
}

# Runtime settings are never stored in a checkpoint.
RUNTIME_DEFAULTS = {
    "device": None,              # None: cuda > mps > cpu
    "load_model": None,          # Checkpoint to load
    "save_best": True,           # Write best model (by validation loss) to best_model_file
    "best_model_file": "best_model.pt",
    "restore_best": True,        # At the end of fit(), keep the best weights instead of the last
    "compile_model": False,      # torch.compile the model for training
    "amp": "auto",               # 'auto' (bf16 on capable CUDA), 'bf16', or None/False
    "seed": None,
}


def _default_device():
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _numericalize(text, token2idx, sos_idx, eos_idx, unk_idx):
    ids = [sos_idx]
    ids.extend(token2idx.get(token, unk_idx) for token in text.split())
    ids.append(eos_idx)
    return torch.tensor(ids, dtype=torch.long)


class EZTransformer:
    def __init__(self, **kwargs):
        defaults = {**ARCH_DEFAULTS, **SETTINGS_DEFAULTS, **RUNTIME_DEFAULTS}
        unknown = sorted(set(kwargs) - set(defaults))
        if unknown:
            raise TypeError(
                f"Unknown argument(s): {', '.join(unknown)}. "
                f"Valid arguments: {', '.join(sorted(defaults))}."
            )
        self._explicit = set(kwargs)
        for key, value in defaults.items():
            setattr(self, self._attr(key), kwargs.get(key, value))
        self.device = self.device or _default_device()

        if self.amp not in ("auto", "bf16", None, False):
            raise ValueError("amp must be 'auto', 'bf16', or None/False.")
        if torch.device(self.device).type == "cuda":
            # Allow TF32 matmuls; a free speedup on Ampere+ GPUs.
            torch.set_float32_matmul_precision("high")
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)

        # Initialize placeholders
        self.model = None
        self.optimizer = None
        self.token2idx = None
        self.idx2token = None
        self.pad_idx = None
        self.sos_idx = None
        self.eos_idx = None
        self.unk_idx = None
        self.best_valid_loss = float("inf")

        if self.load_model:
            self.load_model_from_file(self.load_model)

    @staticmethod
    def _attr(key):
        # The 'optimizer' kwarg is stored as optimizer_name; self.optimizer is the optimizer object.
        return "optimizer_name" if key == "optimizer" else key

    @property
    def _pin_memory(self):
        return torch.device(self.device).type == "cuda"

    @property
    def _amp_dtype(self):
        if self.amp == "bf16":
            return torch.bfloat16
        if self.amp == "auto" and torch.device(self.device).type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return None

    def _autocast(self):
        dtype = self._amp_dtype
        if dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=torch.device(self.device).type, dtype=dtype)

    def _arch_config(self):
        return {key: getattr(self, key) for key in ARCH_DEFAULTS}

    def _settings_config(self):
        return {key: getattr(self, self._attr(key)) for key in SETTINGS_DEFAULTS}

    def _raw_model(self):
        return getattr(self.model, "_orig_mod", self.model)

    def _require_model(self):
        if self.model is None or self.token2idx is None:
            raise RuntimeError("Model is not initialized. Call fit() or load a checkpoint first.")

    def _to_device(self, tensor):
        return tensor.to(self.device, non_blocking=self._pin_memory)

    def fit(self, train_data, valid_data=None, max_epochs=100, print_validation_examples=0, return_history=False):
        if self.token2idx is None:
            self.build_vocab(train_data)
            self.build_model()
            self.optimizer = self.build_optimizer()
        else:
            print("Continuing training with existing model weights.")

        train_loader = self.create_dataloader(train_data, batch_size=self.bts, shuffle=True)
        valid_loader = (
            self.create_dataloader(valid_data, batch_size=self.bts, shuffle=False)
            if valid_data
            else None
        )
        scheduler = self.build_scheduler(max_epochs * len(train_loader))

        criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx, label_smoothing=self.lst)
        history = []
        best_state, best_epoch, run_best_loss = None, None, float("inf")

        for epoch in range(max_epochs):
            self.model.train()
            epoch_loss = torch.zeros((), device=self.device)
            valid_loss = None
            for src_batch, trg_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs}"):
                src_batch = self._to_device(src_batch)
                trg_batch = self._to_device(trg_batch)

                self.optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    output = self.model(src_batch, trg_batch[:, :-1])
                    loss = criterion(
                        output.reshape(-1, output.size(-1)),
                        trg_batch[:, 1:].reshape(-1),
                    )
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cnm)
                self.optimizer.step()
                scheduler.step()

                # Accumulate on device; calling .item() every step would force a sync.
                epoch_loss += loss.detach()

            avg_epoch_loss = epoch_loss.item() / len(train_loader)
            print(f"Epoch {epoch + 1}: Training Loss: {avg_epoch_loss:.6f}")

            if valid_loader:
                valid_loss = self.evaluate(valid_loader)
                print(f"Epoch {epoch + 1}: Validation Loss: {valid_loss:.6f}")

                if self.restore_best and valid_loss < run_best_loss:
                    run_best_loss, best_epoch = valid_loss, epoch + 1
                    best_state = {k: v.detach().clone() for k, v in self._raw_model().state_dict().items()}
                if self.save_best and valid_loss < self.best_valid_loss:
                    self.best_valid_loss = valid_loss
                    self.write_model(self.best_model_file)

            if print_validation_examples > 0 and valid_data:
                self.print_validation_examples(valid_data, n=print_validation_examples)

            history.append({
                "epoch": epoch + 1,
                "train_loss": avg_epoch_loss,
                "val_loss": valid_loss,
            })

        if best_state is not None and best_epoch != max_epochs:
            self._raw_model().load_state_dict(best_state)
            print(f"Restored best weights from epoch {best_epoch} (validation loss {run_best_loss:.6f}).")

        if return_history:
            try:
                import pandas as pd
            except ImportError:
                return history
            return pd.DataFrame(history)

    def build_vocab(self, data):
        tokens = set()
        for src, trg in data:
            tokens.update(src.split())
            tokens.update(trg.split())

        special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
        self.token2idx = {token: idx for idx, token in enumerate(special_tokens)}
        idx = len(self.token2idx)
        for token in sorted(tokens):
            if token not in self.token2idx:
                self.token2idx[token] = idx
                idx += 1
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

        self.pad_idx = self.token2idx["<pad>"]
        self.sos_idx = self.token2idx["<sos>"]
        self.eos_idx = self.token2idx["<eos>"]
        self.unk_idx = self.token2idx["<unk>"]
        self.vocab_size = len(self.token2idx)

    def build_model(self):
        for side, dim, heads in (("Encoder", self.eed, self.eah), ("Decoder", self.ded, self.dah)):
            if dim % heads != 0:
                raise ValueError(f"{side} embedding size ({dim}) must be divisible by its attention heads ({heads}).")
            if self.use_rope and (dim // heads) % 2 != 0:
                raise ValueError(f"{side} head dimension ({dim // heads}) must be even when use_rope=True.")

        self.model = TransformerModel(
            vocab_size=self.vocab_size,
            pad_idx=self.pad_idx,
            enc_dim=self.eed,
            enc_hidden=self.ehs,
            enc_layers=self.enl,
            enc_heads=self.eah,
            dec_dim=self.ded,
            dec_hidden=self.dhs,
            dec_layers=self.dnl,
            dec_heads=self.dah,
            dropout=self.drp,
            use_rope=self.use_rope,
            rope_theta=self.rope_theta,
        ).to(self.device)
        if self.compile_model and hasattr(torch, "compile"):
            # Batch shapes vary, so compile with dynamic shapes to avoid recompiling per batch.
            self.model = torch.compile(self.model, dynamic=True)

    def build_optimizer(self):
        name = self.optimizer_name.lower()
        kwargs = {"lr": self.lrt, "betas": self.adam_betas}
        if name == "adamw":
            opt_cls = optim.AdamW
        elif name == "adam":
            opt_cls = optim.Adam
        else:
            raise ValueError(f"Unsupported optimizer '{self.optimizer_name}'. Supported optimizers: 'adam', 'adamw'.")

        if torch.device(self.device).type == "cuda":
            try:
                return opt_cls(self.model.parameters(), fused=True, **kwargs)
            except (TypeError, RuntimeError):
                pass
        return opt_cls(self.model.parameters(), **kwargs)

    def build_scheduler(self, total_steps):
        # Linear warmup, then cosine decay to 10% of the peak learning rate.
        # Reset lr/betas first: a loaded optimizer state (or an earlier fit) carries its own values.
        for group in self.optimizer.param_groups:
            group["lr"] = self.lrt
            group["betas"] = self.adam_betas
            group.pop("initial_lr", None)
        warmup_steps = max(1, round(self.wup * total_steps))

        def lr_factor(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_factor)

    def create_dataloader(self, data, batch_size, shuffle=True):
        dataset = TranslationDataset(
            data,
            self.token2idx,
            self.sos_idx,
            self.eos_idx,
            self.unk_idx,
            self.pad_idx,
        )
        return DataLoader(
            dataset,
            batch_sampler=LengthBucketSampler(dataset.lengths, batch_size, shuffle=shuffle),
            collate_fn=dataset.collate_fn,
            pin_memory=self._pin_memory,
        )

    def evaluate(self, data_loader):
        """Per-token validation loss (label smoothing included, matching the training objective)."""
        self.model.eval()
        criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx, label_smoothing=self.lst, reduction="sum")
        total_loss = torch.zeros((), device=self.device)
        total_tokens = 0

        with torch.inference_mode(), self._autocast():
            for src_batch, trg_batch in data_loader:
                src_batch = self._to_device(src_batch)
                trg_batch = self._to_device(trg_batch)

                output = self.model(src_batch, trg_batch[:, :-1])
                targets = trg_batch[:, 1:]
                total_loss += criterion(output.reshape(-1, output.size(-1)), targets.reshape(-1))
                total_tokens += (targets != self.pad_idx).sum()

        return (total_loss / total_tokens).item()

    def print_validation_examples(self, valid_data, n=2):
        n = min(n, len(valid_data))
        examples = random.sample(valid_data, n)
        sources = [src for src, _ in examples]
        predictions = self.predict(sources)
        print("\nValidation Examples:")
        for (src, trg), prediction in zip(examples, predictions):
            print(f"Input:     {src}")
            print(f"Target:    {trg}")
            if prediction.strip() == trg.strip():
                print(f"\033[92mPredicted:\033[0m {prediction}\n")
            else:
                print(f"\033[91mPredicted:\033[0m {prediction}\n")

    def _decode_tokens(self, token_ids):
        tokens = []
        for idx in token_ids:
            if idx == self.eos_idx or idx == self.pad_idx:
                break
            tokens.append(self.idx2token[idx])
        return " ".join(tokens)

    def predict(self, test_data, max_len=None, batch_size=256, beam_size=1, length_penalty=1.0, progress=False):
        """Translate a list of whitespace-tokenized strings.

        beam_size=1 is greedy decoding. With beam_size > 1, finished hypotheses are ranked by
        log-probability / length**length_penalty. max_len (or max_pred_len) caps output length;
        if both are None the cap is max(50, 2 * source length + 10) per input.
        """
        self._require_model()
        self.model.eval()
        if not test_data:
            return []

        max_len = self.max_pred_len if max_len is None else max_len
        starts = range(0, len(test_data), batch_size)
        if progress and len(test_data) > batch_size:
            starts = tqdm(starts, desc="Predicting")
        predictions = []
        with torch.inference_mode(), self._autocast():
            for i in starts:
                batch = test_data[i:i + batch_size]
                if beam_size > 1:
                    predictions.extend(self._beam_search_batch(batch, max_len, beam_size, length_penalty))
                else:
                    predictions.extend(self._greedy_batch(batch, max_len))
        return predictions

    def _encode_sources(self, batch, max_len):
        src = nn.utils.rnn.pad_sequence(
            [_numericalize(s, self.token2idx, self.sos_idx, self.eos_idx, self.unk_idx) for s in batch],
            batch_first=True,
            padding_value=self.pad_idx,
        )
        src = self._to_device(src)
        if max_len is None:
            src_lens = (src != self.pad_idx).sum(dim=1) - 2  # exclude <sos>/<eos>
            limits = torch.clamp(2 * src_lens + 10, min=50)
        else:
            limits = torch.full((src.size(0),), max_len, device=src.device)
        memory, memory_mask = self._raw_model().encode(src)
        return memory, memory_mask, limits

    def _next_token_logits(self, model, tokens, memory, memory_mask, cache):
        logits = model.decode(tokens, memory, memory_mask, cache)[:, -1, :].float()
        logits[:, self.pad_idx] = float("-inf")
        logits[:, self.sos_idx] = float("-inf")
        return logits

    def _greedy_batch(self, batch, max_len):
        model = self._raw_model()
        memory, memory_mask, limits = self._encode_sources(batch, max_len)
        batch_size = memory.size(0)
        cache = model.new_cache()

        tokens = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long, device=memory.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=memory.device)
        outputs = []
        for step in range(int(limits.max())):
            # With the KV cache, only the newest token is fed to the decoder.
            logits = self._next_token_logits(model, tokens, memory, memory_mask, cache)
            next_token = logits.argmax(dim=-1).masked_fill(finished, self.pad_idx)
            outputs.append(next_token)
            finished = finished | (next_token == self.eos_idx) | (step + 1 >= limits)
            if torch.all(finished):
                break
            tokens = next_token.unsqueeze(1)

        return [self._decode_tokens(seq) for seq in torch.stack(outputs, dim=1).tolist()]

    def _beam_search_batch(self, batch, max_len, beam_size, length_penalty):
        model = self._raw_model()
        memory, memory_mask, limits = self._encode_sources(batch, max_len)
        batch_size, device = memory.size(0), memory.device
        k = beam_size

        # Hypotheses live in a flat (batch_size * k) dimension, grouped by source.
        memory = memory.repeat_interleave(k, dim=0)
        memory_mask = memory_mask.repeat_interleave(k, dim=0)
        limits = limits.repeat_interleave(k, dim=0)
        cache = model.new_cache()

        scores = torch.full((batch_size, k), float("-inf"), device=device)
        scores[:, 0] = 0.0  # all beams start identical; expand only the first at step 0
        scores = scores.view(-1)
        lengths = torch.zeros(batch_size * k, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size * k, dtype=torch.bool, device=device)
        seqs = torch.empty((batch_size * k, 0), dtype=torch.long, device=device)
        tokens = torch.full((batch_size * k, 1), self.sos_idx, dtype=torch.long, device=device)
        group_offsets = (torch.arange(batch_size, device=device) * k).unsqueeze(1)

        for _ in range(int(limits.max())):
            log_probs = F.log_softmax(self._next_token_logits(model, tokens, memory, memory_mask, cache), dim=-1)
            # A finished hypothesis can only be extended by <pad>, at no cost.
            log_probs[finished] = float("-inf")
            log_probs[finished, self.pad_idx] = 0.0

            vocab_size = log_probs.size(-1)
            candidates = (scores.unsqueeze(1) + log_probs).view(batch_size, k * vocab_size)
            top_scores, top_idx = candidates.topk(k, dim=-1)
            origin = (group_offsets + top_idx // vocab_size).view(-1)
            next_token = (top_idx % vocab_size).view(-1)

            scores = top_scores.view(-1)
            lengths = lengths[origin] + (~finished[origin]).long()
            finished = finished[origin] | (next_token == self.eos_idx)
            finished = finished | (lengths >= limits)
            seqs = torch.cat([seqs[origin], next_token.unsqueeze(1)], dim=1)
            model.reorder_cache(cache, origin)
            if torch.all(finished):
                break
            tokens = next_token.unsqueeze(1)

        normalized = scores / lengths.clamp(min=1).float() ** length_penalty
        best = normalized.view(batch_size, k).argmax(dim=-1)
        best_seqs = seqs.view(batch_size, k, -1)[torch.arange(batch_size, device=device), best]
        return [self._decode_tokens(seq) for seq in best_seqs.tolist()]

    def score(self, test_data, test_outputs, batch_size=256, beam_size=1):
        if len(test_data) != len(test_outputs):
            raise ValueError(f"Got {len(test_data)} inputs but {len(test_outputs)} outputs.")
        predictions = self.predict(test_data, batch_size=batch_size, beam_size=beam_size, progress=True)

        correct = 0
        total = len(test_data)
        total_distance = 0

        for pred, gold in zip(predictions, test_outputs):
            if pred.strip() == gold.strip():
                correct += 1
            total_distance += self.levenshtein_distance(pred.split(), gold.split())

        accuracy = correct / total if total else 0.0
        avg_distance = total_distance / total if total else 0.0

        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Average Levenshtein Distance: {avg_distance:.2f}")
        return accuracy, avg_distance

    def write_model(self, filename="eztransformer_model.pt"):
        model = self._raw_model()
        state = {
            "format_version": CHECKPOINT_FORMAT,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": None if self.optimizer is None else self.optimizer.state_dict(),
            "token2idx": self.token2idx,
            "idx2token": self.idx2token,
            "pad_idx": self.pad_idx,
            "sos_idx": self.sos_idx,
            "eos_idx": self.eos_idx,
            "unk_idx": self.unk_idx,
            "best_valid_loss": self.best_valid_loss,
            "arch": self._arch_config(),
            "settings": self._settings_config(),
        }
        torch.save(state, filename)
        print(f"Model saved to {filename}")

    def load_model_from_file(self, filename):
        state = torch.load(filename, map_location=self.device, weights_only=True)
        if state.get("format_version") != CHECKPOINT_FORMAT:
            raise ValueError(
                f"{filename} was written by an older, incompatible version of eztr "
                "(the model architecture has changed). Please retrain."
            )
        self.token2idx = state["token2idx"]
        self.idx2token = state["idx2token"]
        self.pad_idx = state["pad_idx"]
        self.sos_idx = state["sos_idx"]
        self.eos_idx = state["eos_idx"]
        self.unk_idx = state["unk_idx"]
        self.vocab_size = len(self.token2idx)
        self.best_valid_loss = state.get("best_valid_loss", float("inf"))

        # Architecture always comes from the checkpoint; conflicting explicit values are an error.
        for key, value in state["arch"].items():
            if key in self._explicit and getattr(self, key) != value:
                raise ValueError(
                    f"{key}={getattr(self, key)!r} conflicts with the checkpoint's {key}={value!r}; "
                    "architecture settings are taken from the checkpoint."
                )
            setattr(self, key, value)
        # Training/inference settings: explicitly passed values win over the checkpoint.
        for key, value in state["settings"].items():
            if key not in self._explicit:
                setattr(self, self._attr(key), value)

        self.build_model()
        self._raw_model().load_state_dict(state["model_state_dict"])

        self.optimizer = self.build_optimizer()
        opt_state = state.get("optimizer_state_dict")
        if opt_state:
            self.optimizer.load_state_dict(opt_state)
        print(f"Model loaded from {filename}")

    @staticmethod
    def levenshtein_distance(a, b):
        n, m = len(a), len(b)
        if n > m:
            a, b = b, a
            n, m = m, n

        current_row = list(range(n + 1))
        for i in range(1, m + 1):
            previous_row, current_row = current_row, [i] + [0] * n
            for j in range(1, n + 1):
                insertions = previous_row[j] + 1
                deletions = current_row[j - 1] + 1
                substitutions = previous_row[j - 1] + (a[j - 1] != b[i - 1])
                current_row[j] = min(insertions, deletions, substitutions)
        return current_row[n]


class RMSNorm(nn.Module):
    # Equivalent to nn.RMSNorm (torch >= 2.4); kept local so older PyTorch versions work.
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_float = x.float()
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.weight).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Returns cos/sin tables for rotating query/key heads by their position."""

    def __init__(self, head_dim, theta=10000.0):
        super().__init__()
        inv_freq = theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions):
        angles = torch.outer(positions.float(), self.inv_freq)  # (T, head_dim / 2)
        return angles.cos(), angles.sin()


def apply_rotary(x, cos, sin):
    # x: (B, H, T, head_dim). Rotates dimension pairs (i, i + head_dim/2) by position-dependent
    # angles, so that q_p . k_s depends only on content and the offset s - p.
    x1, x2 = x.chunk(2, dim=-1)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


def sinusoidal_positions(positions, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=positions.device, dtype=torch.float32) / half)
    angles = positions.float().unsqueeze(1) * freqs
    pe = torch.cat((angles.sin(), angles.cos()), dim=-1)
    return F.pad(pe, (0, dim - 2 * half))


class Attention(nn.Module):
    def __init__(self, dim, num_heads, dropout, kv_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(kv_dim or dim, 2 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def _heads(self, x):
        # (B, T, H * head_dim) -> (B, H, T, head_dim)
        return x.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)

    def _keys_values(self, x):
        k, v = self.kv_proj(x).chunk(2, dim=-1)
        return self._heads(k), self._heads(v)

    def forward(self, x, memory=None, mask=None, rope=None, causal=False, cache=None):
        q = self._heads(self.q_proj(x))
        if memory is None:
            # Self-attention, with RoPE applied to queries and keys (never values).
            k, v = self._keys_values(x)
            if rope is not None:
                q, k = apply_rotary(q, *rope), apply_rotary(k, *rope)
            if cache is not None:
                if "k" in cache:
                    # Cached decoding feeds one new token at a time; it may attend to every cached key.
                    k = torch.cat([cache["k"], k], dim=2)
                    v = torch.cat([cache["v"], v], dim=2)
                    causal = False
                cache["k"], cache["v"] = k, v
        else:
            # Cross-attention: no RoPE (source and target positions are unrelated).
            # Encoder keys/values are computed once per decode and cached.
            if cache is not None and "k" in cache:
                k, v = cache["k"], cache["v"]
            else:
                k, v = self._keys_values(memory)
                if cache is not None:
                    cache["k"], cache["v"] = k, v

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )
        return self.out_proj(out.transpose(1, 2).flatten(2))


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_size):
        super().__init__()
        self.w_in = nn.Linear(dim, 2 * hidden_size, bias=False)
        self.w_out = nn.Linear(hidden_size, dim, bias=False)

    def forward(self, x):
        gate, value = self.w_in(x).chunk(2, dim=-1)
        return self.w_out(F.silu(gate) * value)


class EncoderLayer(nn.Module):
    def __init__(self, dim, hidden_size, num_heads, dropout):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, rope):
        x = x + self.dropout(self.attn(self.attn_norm(x), mask=mask, rope=rope))
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class DecoderLayer(nn.Module):
    def __init__(self, dim, hidden_size, num_heads, enc_dim, dropout):
        super().__init__()
        self.self_attn_norm = RMSNorm(dim)
        self.self_attn = Attention(dim, num_heads, dropout)
        self.cross_attn_norm = RMSNorm(dim)
        self.cross_attn = Attention(dim, num_heads, dropout, kv_dim=enc_dim)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, memory_mask, rope, cache=None):
        self_cache = None if cache is None else cache["self"]
        cross_cache = None if cache is None else cache["cross"]
        x = x + self.dropout(self.self_attn(self.self_attn_norm(x), rope=rope, causal=True, cache=self_cache))
        x = x + self.dropout(self.cross_attn(self.cross_attn_norm(x), memory=memory, mask=memory_mask, cache=cross_cache))
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class TransformerModel(nn.Module):
    """Pre-norm encoder-decoder Transformer with RMSNorm, SwiGLU, RoPE, and a decoder KV cache."""

    def __init__(self, vocab_size, pad_idx, enc_dim, enc_hidden, enc_layers, enc_heads,
                 dec_dim, dec_hidden, dec_layers, dec_heads, dropout, use_rope=True, rope_theta=10000.0):
        super().__init__()
        self.pad_idx = pad_idx
        self.use_rope = use_rope

        self.src_embedding = nn.Embedding(vocab_size, enc_dim)
        self.trg_embedding = nn.Embedding(vocab_size, dec_dim)
        # Embeddings are scaled by sqrt(dim) on input, so initialize them with std 1/sqrt(dim).
        nn.init.normal_(self.src_embedding.weight, std=enc_dim ** -0.5)
        nn.init.normal_(self.trg_embedding.weight, std=dec_dim ** -0.5)
        self.emb_dropout = nn.Dropout(dropout)

        if use_rope:
            self.enc_rope = RotaryEmbedding(enc_dim // enc_heads, rope_theta)
            self.dec_rope = RotaryEmbedding(dec_dim // dec_heads, rope_theta)

        self.encoder_layers = nn.ModuleList(
            EncoderLayer(enc_dim, enc_hidden, enc_heads, dropout) for _ in range(enc_layers)
        )
        self.decoder_layers = nn.ModuleList(
            DecoderLayer(dec_dim, dec_hidden, dec_heads, enc_dim, dropout) for _ in range(dec_layers)
        )
        self.enc_norm = RMSNorm(enc_dim)
        self.dec_norm = RMSNorm(dec_dim)

        # Output projection is tied to the target embedding.
        self.fc_out = nn.Linear(dec_dim, vocab_size, bias=False)
        self.fc_out.weight = self.trg_embedding.weight

    def _embed(self, embedding, tokens, rotary, offset=0):
        x = embedding(tokens) * math.sqrt(embedding.embedding_dim)
        positions = torch.arange(offset, offset + tokens.size(1), device=tokens.device)
        if self.use_rope:
            rope = rotary(positions)
        else:
            rope = None
            x = x + sinusoidal_positions(positions, embedding.embedding_dim).to(x.dtype)
        return self.emb_dropout(x), rope

    def encode(self, src):
        x, rope = self._embed(self.src_embedding, src, self.enc_rope if self.use_rope else None)
        # Boolean mask, True = attend; broadcast as (B, 1, 1, S) over heads and queries.
        src_mask = (src != self.pad_idx)[:, None, None, :]
        for layer in self.encoder_layers:
            x = layer(x, src_mask, rope)
        return self.enc_norm(x), src_mask

    def decode(self, trg, memory, memory_mask, cache=None):
        offset = 0 if cache is None else cache["len"]
        x, rope = self._embed(self.trg_embedding, trg, self.dec_rope if self.use_rope else None, offset)
        # No target padding mask: targets are right-padded and self-attention is causal, so real
        # positions never attend to pads (pad positions are ignored by the loss).
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x, memory, memory_mask, rope, None if cache is None else cache["layers"][i])
        if cache is not None:
            cache["len"] += trg.size(1)
        return self.fc_out(self.dec_norm(x))

    def forward(self, src, trg):
        memory, src_mask = self.encode(src)
        return self.decode(trg, memory, src_mask)

    def new_cache(self):
        return {"len": 0, "layers": [{"self": {}, "cross": {}} for _ in self.decoder_layers]}

    @staticmethod
    def reorder_cache(cache, indices):
        # Beam search: hypotheses only move within their own source's group, so the
        # cross-attention cache (identical across a group) needs no reordering.
        for layer_cache in cache["layers"]:
            self_cache = layer_cache["self"]
            if self_cache:
                self_cache["k"] = self_cache["k"].index_select(0, indices)
                self_cache["v"] = self_cache["v"].index_select(0, indices)


class LengthBucketSampler(torch.utils.data.Sampler):
    """Batches examples of similar length to reduce padding.

    Shuffles, sorts by length within chunks of `chunk_batches` batches, then shuffles batch order.
    """

    def __init__(self, lengths, batch_size, shuffle=True, chunk_batches=50):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.chunk_size = batch_size * chunk_batches

    def __iter__(self):
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            random.shuffle(indices)
        batches = []
        for start in range(0, len(indices), self.chunk_size):
            chunk = sorted(indices[start:start + self.chunk_size], key=self.lengths.__getitem__)
            batches.extend(chunk[i:i + self.batch_size] for i in range(0, len(chunk), self.batch_size))
        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        n = len(self.lengths)
        full_chunks, remainder = divmod(n, self.chunk_size)
        return full_chunks * math.ceil(self.chunk_size / self.batch_size) + math.ceil(remainder / self.batch_size)


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, data, token2idx, sos_idx, eos_idx, unk_idx, pad_idx):
        self.pad_idx = pad_idx
        self.src = []
        self.trg = []
        for src, trg in data:
            self.src.append(_numericalize(src, token2idx, sos_idx, eos_idx, unk_idx))
            self.trg.append(_numericalize(trg, token2idx, sos_idx, eos_idx, unk_idx))
        self.lengths = [len(s) + len(t) for s, t in zip(self.src, self.trg)]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.trg[idx]

    def collate_fn(self, batch):
        src_batch, trg_batch = zip(*batch)
        src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=self.pad_idx)
        trg_padded = nn.utils.rnn.pad_sequence(trg_batch, batch_first=True, padding_value=self.pad_idx)
        return src_padded, trg_padded
