import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
import math
import pandas as pd


class EZTransformer:
    def __init__(self, **kwargs):
        # Set default hyperparameters
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.eed = kwargs.get('eed', 256)        # Encoder embedding dimension
        self.ehs = kwargs.get('ehs', 1024)       # Encoder hidden size
        self.enl = kwargs.get('enl', 4)          # Encoder number of layers
        self.eah = kwargs.get('eah', 4)          # Encoder attention heads
        self.ded = kwargs.get('ded', 256)        # Decoder embedding dimension
        self.dhs = kwargs.get('dhs', 1024)       # Decoder hidden size
        self.dnl = kwargs.get('dnl', 4)          # Decoder number of layers
        self.dah = kwargs.get('dah', 4)          # Decoder attention heads
        self.drp = kwargs.get('drp', 0.3)        # Dropout
        self.bts = kwargs.get('bts', 800)        # Batch size
        self.lrt = kwargs.get('lrt', 0.001)      # Learning rate
        self.lst = kwargs.get('lst', 0.1)        # Label smoothing
        self.cnm = kwargs.get('cnm', 1.0)        # Clip norm
        self.optimizer_name = kwargs.get('optimizer', 'adam')
        self.adam_betas = kwargs.get('adam_betas', (0.9, 0.999))
        self.save_best = kwargs.get('save_best', True)
        self.load_model = kwargs.get('load_model', None)
        self.use_rope = kwargs.get('use_rope', True)

        # Initialize placeholders
        self.model = None
        self.optimizer = None
        self.token2idx = None
        self.idx2token = None
        self.pad_idx = None
        self.sos_idx = None
        self.eos_idx = None
        self.unk_idx = None
        self.best_valid_loss = float('inf')

        if self.load_model:
            self.load_model_from_file(self.load_model)

    def fit(self, train_data, valid_data=None, max_epochs=100, print_validation_examples=0, return_history=False):
        # Build vocabulary from train_data if not already built
        if self.token2idx is None:
            self.build_vocab(train_data)
            self.build_model()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lrt, betas=self.adam_betas)
        else:
            print("Continuing training with existing model weights.")

        # Prepare data loaders
        train_loader = self.create_dataloader(train_data, batch_size=self.bts)
        if valid_data:
            valid_loader = self.create_dataloader(valid_data, batch_size=self.bts)
        else:
            valid_loader = None

        # Define loss function
        criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx, label_smoothing=self.lst)

        # Define the history dataframe
        training_history = pd.DataFrame(columns = ["epoch", "train_loss", "val_loss"])
        
        # Training loop
        for epoch in range(max_epochs):
            # Training
            self.model.train()
            epoch_loss = 0
            for src_batch, trg_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}"):
                src_batch = src_batch.to(self.device)
                trg_batch = trg_batch.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(src_batch, trg_batch[:, :-1])

                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg_batch = trg_batch[:, 1:].contiguous().view(-1)

                loss = criterion(output, trg_batch)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cnm)
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_loader)

            print(f"Epoch {epoch+1}: Training Loss: {avg_epoch_loss:.6f}")

            # Validation
            if valid_loader:
                valid_loss = self.evaluate(valid_loader, criterion)
                print(f"Epoch {epoch+1}: Validation Loss: {valid_loss:.6f}")

                # Save the best model
                if self.save_best and valid_loss < self.best_valid_loss:
                    self.best_valid_loss = valid_loss
                    self.write_model('best_model.pt')

            # Print validation examples
            if print_validation_examples > 0 and valid_data:
                self.print_validation_examples(valid_data, n=print_validation_examples)
            
            training_history.loc[epoch, "epoch"] = epoch + 1
            training_history.loc[epoch, "train_loss"] = avg_epoch_loss
            training_history.loc[epoch, "val_loss"] = valid_loss

        if return_history:
            return training_history

    def build_vocab(self, data):
        tokens = set()
        for src, trg in data:
            tokens.update(src.split())
            tokens.update(trg.split())

        special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.token2idx = {token: idx for idx, token in enumerate(special_tokens)}
        idx = len(self.token2idx)
        for token in sorted(tokens):
            if token not in self.token2idx:
                self.token2idx[token] = idx
                idx += 1
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

        self.pad_idx = self.token2idx['<pad>']
        self.sos_idx = self.token2idx['<sos>']
        self.eos_idx = self.token2idx['<eos>']
        self.unk_idx = self.token2idx['<unk>']
        self.vocab_size = len(self.token2idx)

    def build_model(self):
        self.model = TransformerModel(
            vocab_size=self.vocab_size,
            emb_size=self.eed,
            hidden_size=self.ehs,
            num_layers=self.enl,
            num_heads=self.eah,
            dec_emb_size=self.ded,
            dec_hidden_size=self.dhs,
            dec_num_layers=self.dnl,
            dec_num_heads=self.dah,
            dropout=self.drp,
            pad_idx=self.pad_idx,
            use_rope=self.use_rope
        ).to(self.device)

    def create_dataloader(self, data, batch_size):
        dataset = TranslationDataset(
            data,
            self.token2idx,
            self.sos_idx,
            self.eos_idx,
            self.unk_idx,
            self.pad_idx
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    def evaluate(self, data_loader, criterion):
        self.model.eval()
        epoch_loss = 0

        with torch.no_grad():
            for src_batch, trg_batch in data_loader:
                src_batch = src_batch.to(self.device)
                trg_batch = trg_batch.to(self.device)

                output = self.model(src_batch, trg_batch[:, :-1])
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg_batch = trg_batch[:, 1:].contiguous().view(-1)

                loss = criterion(output, trg_batch)
                epoch_loss += loss.item()

        avg_loss = epoch_loss / len(data_loader)
        return avg_loss

    def print_validation_examples(self, valid_data, n=2):
        examples = random.sample(valid_data, n)
        print("\nValidation Examples:")
        for src, trg in examples:
            prediction = self.predict([src])[0]
            print(f"Input:     {src}")
            print(f"Target:    {trg}")
            if prediction.strip() == trg.strip():
                # "Predicted:" in green, prediction in default color
                print(f"\033[92mPredicted:\033[0m {prediction}\n")
            else:
                # "Predicted:" in red, prediction in default color
                print(f"\033[91mPredicted:\033[0m {prediction}\n")

    def predict(self, test_data):
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for src in test_data:
                src_indices = [self.token2idx.get(token, self.unk_idx) for token in src.split()]
                src_tensor = torch.LongTensor([self.sos_idx] + src_indices + [self.eos_idx]).unsqueeze(0).to(self.device)

                max_len = 50  # Maximum prediction length
                trg_indices = [self.sos_idx]

                for _ in range(max_len):
                    trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(self.device)
                    output = self.model(src_tensor, trg_tensor)
                    next_token = output.argmax(2)[:, -1].item()
                    trg_indices.append(next_token)
                    if next_token == self.eos_idx:
                        break

                tokens = [self.idx2token[idx] for idx in trg_indices[1:-1]]
                predictions.append(' '.join(tokens))

        return predictions
    
    def score(self, test_data, test_outputs, batch_size=64):
        predictions = []
        # Show progress over the batches we feed into predict
        for i in tqdm(range(0, len(test_data), batch_size), desc="Scoring"):
            batch = test_data[i:i+batch_size]
            batch_predictions = self.predict(batch)
            predictions.extend(batch_predictions)

        correct = 0
        total = len(test_data)
        total_distance = 0

        for pred, gold in zip(predictions, test_outputs):
            if pred.strip() == gold.strip():
                correct += 1
            distance = self.levenshtein_distance(pred.split(), gold.split())
            total_distance += distance

        accuracy = correct / total
        avg_distance = total_distance / total

        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Average Levenshtein Distance: {avg_distance:.2f}")
        return accuracy, avg_distance
    

    def write_model(self, filename='eztransformer_model.pt'):
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'token2idx': self.token2idx,
            'idx2token': self.idx2token,
            'pad_idx': self.pad_idx,
            'sos_idx': self.sos_idx,
            'eos_idx': self.eos_idx,
            'unk_idx': self.unk_idx,
            'best_valid_loss': self.best_valid_loss,
        }
        torch.save(state, filename)
        print(f"Model saved to {filename}")

    def load_model_from_file(self, filename):
        state = torch.load(filename, map_location=self.device)
        self.token2idx = state['token2idx']
        self.idx2token = state['idx2token']
        self.pad_idx = state['pad_idx']
        self.sos_idx = state['sos_idx']
        self.eos_idx = state['eos_idx']
        self.unk_idx = state['unk_idx']
        self.vocab_size = len(self.token2idx)
        self.best_valid_loss = state.get('best_valid_loss', float('inf'))

        self.build_model()
        self.model.load_state_dict(state['model_state_dict'])

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lrt, betas=self.adam_betas)
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
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
                 dec_emb_size, dec_hidden_size, dec_num_layers, dec_num_heads,
                 dropout, pad_idx, use_rope=False):
        super(TransformerModel, self).__init__()

        self.pad_idx = pad_idx
        self.use_rope = use_rope
        self.src_embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.trg_embedding = nn.Embedding(vocab_size, dec_emb_size, padding_idx=pad_idx)

        if self.use_rope:
            self.pos_encoder = RoPEEncoding(emb_size)
            self.pos_decoder = RoPEEncoding(dec_emb_size)
        else:
            self.pos_encoder = PositionalEncoding(emb_size, dropout)
            self.pos_decoder = PositionalEncoding(dec_emb_size, dropout)

        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=dec_num_layers,
            dim_feedforward=hidden_size,
            dropout=dropout
        )

        self.fc_out = nn.Linear(emb_size, vocab_size)

    def forward(self, src, trg):
        src_mask = None
        trg_mask = self.generate_square_subsequent_mask(trg.size(1)).to(trg.device)

        src_emb = self.src_embedding(src) * math.sqrt(self.src_embedding.embedding_dim)
        src_emb = self.pos_encoder(src_emb)

        trg_emb = self.trg_embedding(trg) * math.sqrt(self.trg_embedding.embedding_dim)
        trg_emb = self.pos_decoder(trg_emb)

        src_key_padding_mask = (src == self.pad_idx)
        trg_key_padding_mask = (trg == self.pad_idx)

        output = self.transformer(
            src_emb.permute(1, 0, 2),
            trg_emb.permute(1, 0, 2),
            src_mask=src_mask,
            tgt_mask=trg_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=trg_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        output = self.fc_out(output.permute(1, 0, 2))
        return output

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(maxlen, emb_size)
        position = torch.arange(0, maxlen, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))

        pe[:, 0::2] = torch.sin(position * div_term)
        if emb_size % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)

class RoPEEncoding(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.emb_size = emb_size
        # Precompute frequency base
        theta = 100.0
        half_dim = emb_size // 2
        freq = torch.exp(-math.log(theta) * torch.arange(0, half_dim, dtype=torch.float) / half_dim)
        self.register_buffer('freq', freq)

    def forward(self, x):
        # x: [batch, seq_len, emb_size]
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(1)  # [seq_len, 1]
        freqs = self.freq.unsqueeze(0)  # [1, half_dim]

        angles = positions * freqs  # [seq_len, half_dim]
        sin = torch.sin(angles)
        cos = torch.cos(angles)

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_odd * cos + x_even * sin

        x = torch.zeros_like(x)
        x[..., ::2] = x_rotated_even
        x[..., 1::2] = x_rotated_odd
        return x

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, data, token2idx, sos_idx, eos_idx, unk_idx, pad_idx):
        self.data = data
        self.token2idx = token2idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.unk_idx = unk_idx
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, trg = self.data[idx]
        src_indices = [self.token2idx.get(token, self.unk_idx) for token in src.split()]
        trg_indices = [self.token2idx.get(token, self.unk_idx) for token in trg.split()]
        return torch.LongTensor(src_indices), torch.LongTensor(trg_indices)

    def collate_fn(self, batch):
        src_batch, trg_batch = zip(*batch)
        src_batch = [torch.cat([torch.tensor([self.sos_idx]), seq, torch.tensor([self.eos_idx])]) for seq in src_batch]
        trg_batch = [torch.cat([torch.tensor([self.sos_idx]), seq, torch.tensor([self.eos_idx])]) for seq in trg_batch]

        src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=self.pad_idx)
        trg_padded = nn.utils.rnn.pad_sequence(trg_batch, batch_first=True, padding_value=self.pad_idx)

        return src_padded, trg_padded
