import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def _default_device():
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class EZTransformer:
    def __init__(self, **kwargs):
        # Set default hyperparameters
        self.device = kwargs.get("device", _default_device())
        self.eed = kwargs.get("eed", 256)        # Encoder embedding dimension
        self.ehs = kwargs.get("ehs", 1024)       # Encoder hidden size
        self.enl = kwargs.get("enl", 4)          # Encoder number of layers
        self.eah = kwargs.get("eah", 4)          # Encoder attention heads
        self.ded = kwargs.get("ded", 256)        # Decoder embedding dimension
        self.dhs = kwargs.get("dhs", 1024)       # Decoder hidden size
        self.dnl = kwargs.get("dnl", 4)          # Decoder number of layers
        self.dah = kwargs.get("dah", 4)          # Decoder attention heads
        self.drp = kwargs.get("drp", 0.3)        # Dropout
        self.bts = kwargs.get("bts", 800)        # Batch size
        self.lrt = kwargs.get("lrt", 0.001)      # Learning rate
        self.lst = kwargs.get("lst", 0.1)        # Label smoothing
        self.cnm = kwargs.get("cnm", 1.0)        # Clip norm
        self.optimizer_name = kwargs.get("optimizer", "adam")
        self.adam_betas = kwargs.get("adam_betas", (0.9, 0.999))
        self.save_best = kwargs.get("save_best", True)
        self.load_model = kwargs.get("load_model", None)
        self.use_rope = kwargs.get("use_rope", True)
        self.compile_model = kwargs.get("compile_model", False)
        self.max_pred_len = kwargs.get("max_pred_len", 50)

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

    @property
    def _pin_memory(self):
        return torch.device(self.device).type == "cuda"

    def _arch_config(self):
        return {
            "eed": self.eed,
            "ehs": self.ehs,
            "enl": self.enl,
            "eah": self.eah,
            "ded": self.ded,
            "dhs": self.dhs,
            "dnl": self.dnl,
            "dah": self.dah,
            "drp": self.drp,
            "use_rope": self.use_rope,
            "lrt": self.lrt,
            "optimizer_name": self.optimizer_name,
            "adam_betas": self.adam_betas,
            "max_pred_len": self.max_pred_len,
        }

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

        criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx, label_smoothing=self.lst)
        history = []

        for epoch in range(max_epochs):
            self.model.train()
            epoch_loss = 0
            valid_loss = None
            for src_batch, trg_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs}"):
                src_batch = self._to_device(src_batch)
                trg_batch = self._to_device(trg_batch)

                self.optimizer.zero_grad(set_to_none=True)
                output = self.model(src_batch, trg_batch[:, :-1])

                loss = criterion(
                    output.reshape(-1, output.size(-1)),
                    trg_batch[:, 1:].reshape(-1),
                )
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cnm)
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1}: Training Loss: {avg_epoch_loss:.6f}")

            if valid_loader:
                valid_loss = self.evaluate(valid_loader, criterion)
                print(f"Epoch {epoch + 1}: Validation Loss: {valid_loss:.6f}")

                if self.save_best and valid_loss < self.best_valid_loss:
                    self.best_valid_loss = valid_loss
                    self.write_model("best_model.pt")

            if print_validation_examples > 0 and valid_data:
                self.print_validation_examples(valid_data, n=print_validation_examples)

            history.append({
                "epoch": epoch + 1,
                "train_loss": avg_epoch_loss,
                "val_loss": valid_loss,
            })

        if return_history:
            import pandas as pd
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
        # nn.Transformer uses a single d_model across encoder/decoder, so these must match.
        if self.ded != self.eed:
            raise ValueError("Decoder embedding size (ded) must match encoder embedding size (eed) for nn.Transformer.")
        if self.dah != self.eah:
            raise ValueError("Decoder attention heads (dah) must match encoder attention heads (eah) for nn.Transformer.")
        if self.dhs != self.ehs:
            raise ValueError("Decoder hidden size (dhs) must match encoder hidden size (ehs) for nn.Transformer.")

        self.model = TransformerModel(
            vocab_size=self.vocab_size,
            emb_size=self.eed,
            hidden_size=self.ehs,
            num_layers=self.enl,
            num_heads=self.eah,
            dec_num_layers=self.dnl,
            dropout=self.drp,
            pad_idx=self.pad_idx,
            use_rope=self.use_rope,
        ).to(self.device)
        if self.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

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
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=dataset.collate_fn,
            pin_memory=self._pin_memory,
        )

    def evaluate(self, data_loader, criterion):
        self.model.eval()
        epoch_loss = 0

        with torch.inference_mode():
            for src_batch, trg_batch in data_loader:
                src_batch = self._to_device(src_batch)
                trg_batch = self._to_device(trg_batch)

                output = self.model(src_batch, trg_batch[:, :-1])
                loss = criterion(
                    output.reshape(-1, output.size(-1)),
                    trg_batch[:, 1:].reshape(-1),
                )
                epoch_loss += loss.item()

        return epoch_loss / len(data_loader)

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

    def _numericalize(self, text):
        ids = [self.sos_idx]
        ids.extend(self.token2idx.get(token, self.unk_idx) for token in text.split())
        ids.append(self.eos_idx)
        return torch.tensor(ids, dtype=torch.long)

    def _decode_tokens(self, token_ids):
        tokens = []
        for idx in token_ids:
            if idx == self.eos_idx or idx == self.pad_idx:
                break
            tokens.append(self.idx2token[idx])
        return " ".join(tokens)

    def predict(self, test_data, max_len=None, batch_size=64):
        self._require_model()
        self.model.eval()
        if not test_data:
            return []

        max_len = self.max_pred_len if max_len is None else max_len
        predictions = []
        with torch.inference_mode():
            for i in range(0, len(test_data), batch_size):
                predictions.extend(self._predict_batch(test_data[i:i + batch_size], max_len))
        return predictions

    def _predict_batch(self, batch, max_len):
        # Encode each source once, then greedily decode the whole batch in parallel.
        src = nn.utils.rnn.pad_sequence(
            [self._numericalize(src) for src in batch],
            batch_first=True,
            padding_value=self.pad_idx,
        )
        src = self._to_device(src)
        model = self._raw_model()
        memory, memory_pad_mask = model.encode(src)

        batch_size = src.size(0)
        ys = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long, device=src.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)

        for _ in range(max_len):
            logits = model.decode(ys, memory, memory_pad_mask)[:, -1, :]
            logits[:, self.pad_idx] = float("-inf")
            next_token = logits.argmax(dim=-1)
            next_token = next_token.masked_fill(finished, self.pad_idx)
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == self.eos_idx)
            if torch.all(finished):
                break

        return [self._decode_tokens(seq) for seq in ys[:, 1:].tolist()]

    def score(self, test_data, test_outputs, batch_size=64):
        predictions = []
        indices = range(0, len(test_data), batch_size)
        if len(test_data) > batch_size:
            indices = tqdm(indices, desc="Scoring")
        for i in indices:
            predictions.extend(self.predict(test_data[i:i + batch_size], batch_size=batch_size))

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
        }
        torch.save(state, filename)
        print(f"Model saved to {filename}")

    def load_model_from_file(self, filename):
        state = torch.load(filename, map_location=self.device, weights_only=False)
        self.token2idx = state["token2idx"]
        self.idx2token = state["idx2token"]
        self.pad_idx = state["pad_idx"]
        self.sos_idx = state["sos_idx"]
        self.eos_idx = state["eos_idx"]
        self.unk_idx = state["unk_idx"]
        self.vocab_size = len(self.token2idx)
        self.best_valid_loss = state.get("best_valid_loss", float("inf"))

        for key, value in state.get("arch", {}).items():
            if key in self._arch_config():
                setattr(self, key, value)

        self.build_model()
        state_dict = state["model_state_dict"]
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        self._raw_model().load_state_dict(state_dict)

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


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, num_heads,
                 dec_num_layers, dropout, pad_idx, use_rope=False):
        super().__init__()

        self.pad_idx = pad_idx
        self.use_rope = use_rope
        self.src_embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.trg_embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)

        if self.use_rope:
            self.pos_encoder = RoPEEncoding(emb_size)
            self.pos_decoder = RoPEEncoding(emb_size)
        else:
            self.pos_encoder = PositionalEncoding(emb_size, dropout)
            self.pos_decoder = PositionalEncoding(emb_size, dropout)

        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=dec_num_layers,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True,
        )
        # Nested-tensor conversion dominates on short character-level sequences.
        encoder = self.transformer.encoder
        if hasattr(encoder, "enable_nested_tensor"):
            encoder.enable_nested_tensor = False
        if hasattr(encoder, "use_nested_tensor"):
            encoder.use_nested_tensor = False

        self.fc_out = nn.Linear(emb_size, vocab_size)

    def encode(self, src):
        src_emb = self.src_embedding(src) * math.sqrt(self.src_embedding.embedding_dim)
        src_emb = self.pos_encoder(src_emb)
        src_key_padding_mask = src == self.pad_idx
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        return memory, src_key_padding_mask

    def decode(self, trg, memory, memory_key_padding_mask):
        trg_emb = self.trg_embedding(trg) * math.sqrt(self.trg_embedding.embedding_dim)
        trg_emb = self.pos_decoder(trg_emb)
        trg_key_padding_mask = trg == self.pad_idx
        # Bool causal mask matches padding-mask dtype; tgt_is_causal is the SDPA hint.
        tgt_mask = torch.triu(
            torch.ones(trg.size(1), trg.size(1), dtype=torch.bool, device=trg.device),
            diagonal=1,
        )
        output = self.transformer.decoder(
            trg_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_is_causal=True,
            tgt_key_padding_mask=trg_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.fc_out(output)

    def forward(self, src, trg):
        memory, src_key_padding_mask = self.encode(src)
        return self.decode(trg, memory, src_key_padding_mask)


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(maxlen, emb_size)
        position = torch.arange(0, maxlen, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))

        pe[:, 0::2] = torch.sin(position * div_term)
        if emb_size % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(dtype=x.dtype)
        return self.dropout(x)


class RoPEEncoding(nn.Module):
    def __init__(self, emb_size, maxlen=5000, theta=100.0):
        super().__init__()
        self.emb_size = emb_size
        self.rot_dim = (emb_size // 2) * 2
        half_dim = self.rot_dim // 2
        if half_dim == 0:
            self.register_buffer("cos", torch.empty(1, 1, 0))
            self.register_buffer("sin", torch.empty(1, 1, 0))
            return

        inv_freq = torch.exp(-math.log(theta) * torch.arange(half_dim, dtype=torch.float32) / half_dim)
        angles = torch.arange(maxlen, dtype=torch.float32).unsqueeze(1) * inv_freq
        self.register_buffer("cos", torch.cos(angles).unsqueeze(0))
        self.register_buffer("sin", torch.sin(angles).unsqueeze(0))

    def forward(self, x):
        if self.rot_dim == 0:
            return x

        seq_len = x.size(1)
        x_rot = x[..., :self.rot_dim]
        x_even = x_rot[..., ::2]
        x_odd = x_rot[..., 1::2]
        cos = self.cos[:, :seq_len].to(dtype=x.dtype)
        sin = self.sin[:, :seq_len].to(dtype=x.dtype)

        rotated = torch.stack((x_even * cos - x_odd * sin, x_odd * cos + x_even * sin), dim=-1).flatten(-2)
        if self.rot_dim != self.emb_size:
            rotated = torch.cat([rotated, x[..., self.rot_dim:]], dim=-1)
        return rotated


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, data, token2idx, sos_idx, eos_idx, unk_idx, pad_idx):
        self.pad_idx = pad_idx
        self.src = []
        self.trg = []
        for src, trg in data:
            self.src.append(self._numericalize(src, token2idx, sos_idx, eos_idx, unk_idx))
            self.trg.append(self._numericalize(trg, token2idx, sos_idx, eos_idx, unk_idx))

    @staticmethod
    def _numericalize(text, token2idx, sos_idx, eos_idx, unk_idx):
        ids = [sos_idx]
        ids.extend(token2idx.get(token, unk_idx) for token in text.split())
        ids.append(eos_idx)
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.trg[idx]

    def collate_fn(self, batch):
        src_batch, trg_batch = zip(*batch)
        src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=self.pad_idx)
        trg_padded = nn.utils.rnn.pad_sequence(trg_batch, batch_first=True, padding_value=self.pad_idx)
        return src_padded, trg_padded
