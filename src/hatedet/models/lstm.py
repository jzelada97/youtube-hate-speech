"""Red neuronal recurrente (BiLSTM/BiGRU) en PyTorch con interfaz de scikit-learn (Nivel Avanzado).

`LSTMClassifier` implementa fit / predict_proba, asi que se evalua con el mismo protocolo (CV pareada, mismas
metricas) y se puede servir igual que el resto de modelos. Con ~800 comentarios de entrenamiento una red recurrente
entrenada desde cero tiene poco margen: se regulariza fuerte (dropout de embeddings y de palabras, decaimiento
de pesos, parada temprana con un subconjunto de validacion) y se compara honestamente contra TF-IDF.
"""

import copy
import random
from collections import Counter

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from torch import nn

from hatedet.nlp.cleaner import TextCleaner
from hatedet.nlp.group_masker import GroupMasker

PAD, UNK = 0, 1


class _Net(nn.Module):
    def __init__(self, vocab, emb_dim, hidden, cell, dropout, emb_dropout, pretrained=None):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=PAD)
        if pretrained is not None:
            self.emb.weight.data.copy_(torch.as_tensor(pretrained))
        self.emb_drop = nn.Dropout(emb_dropout)
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(4 * hidden, 1)

    def forward(self, x):
        mask = (x != PAD).unsqueeze(-1)
        h, _ = self.rnn(self.emb_drop(self.emb(x)))
        mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        mx = h.masked_fill(~mask, -1e9).max(1).values
        return self.out(self.drop(torch.cat([mean, mx], dim=1))).squeeze(-1)


class LSTMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, cell="lstm", emb_dim=64, hidden=48, dropout=0.5, emb_dropout=0.3, word_dropout=0.1,
                 lr=2e-3, weight_decay=1e-2, max_epochs=40, patience=5, batch_size=32, max_len=80,
                 max_vocab=8000, min_freq=2, val_fraction=0.15, seed=42, pretrained=None, freeze_embeddings=False):
        self.cell, self.emb_dim, self.hidden, self.dropout, self.emb_dropout = cell, emb_dim, hidden, dropout, emb_dropout
        self.word_dropout, self.lr, self.weight_decay = word_dropout, lr, weight_decay
        self.max_epochs, self.patience, self.batch_size, self.max_len = max_epochs, patience, batch_size, max_len
        self.max_vocab, self.min_freq, self.val_fraction, self.seed = max_vocab, min_freq, val_fraction, seed
        self.pretrained, self.freeze_embeddings = pretrained, freeze_embeddings

    def _tokens(self, texts):
        prep = GroupMasker().transform(TextCleaner().transform(texts))
        return [t.split() for t in prep]

    def _encode(self, token_lists):
        ids = np.zeros((len(token_lists), self.max_len), dtype=np.int64)
        for i, toks in enumerate(token_lists):
            row = [self.vocab_.get(t, UNK) for t in toks[: self.max_len]]
            ids[i, : len(row)] = row
        return torch.as_tensor(ids)

    def _seed(self):
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)

    def fit(self, X, y):
        self._seed()
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        tokens, y = self._tokens(list(X)), np.asarray(y).astype(np.float32)
        tr_idx, va_idx = train_test_split(np.arange(len(y)), test_size=self.val_fraction, stratify=y,
                                          random_state=self.seed)
        counts = Counter(t for i in tr_idx for t in tokens[i])
        vocab = [w for w, c in counts.most_common(self.max_vocab) if c >= self.min_freq]
        self.vocab_ = {w: i + 2 for i, w in enumerate(vocab)}
        xtr, xva = self._encode([tokens[i] for i in tr_idx]), self._encode([tokens[i] for i in va_idx])
        ytr, yva = torch.as_tensor(y[tr_idx]), torch.as_tensor(y[va_idx])

        emb = None
        if self.pretrained is not None:
            emb = self.pretrained(self.vocab_, self.emb_dim)
        self.net_ = _Net(len(self.vocab_) + 2, self.emb_dim, self.hidden, self.cell, self.dropout,
                         self.emb_dropout, emb)
        if emb is not None and self.freeze_embeddings:
            self.net_.emb.weight.requires_grad_(False)
        pos_weight = torch.tensor(float((len(ytr) - ytr.sum().item()) / max(1.0, ytr.sum().item())))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW([q for q in self.net_.parameters() if q.requires_grad], lr=self.lr,
                                weight_decay=self.weight_decay)

        best, best_state, bad = float("inf"), None, 0
        self.history_ = []
        for epoch in range(self.max_epochs):
            self.net_.train()
            perm = torch.randperm(len(ytr))
            for s in range(0, len(perm), self.batch_size):
                b = perm[s : s + self.batch_size]
                xb = xtr[b].clone()
                if self.word_dropout:
                    xb[(torch.rand(xb.shape) < self.word_dropout) & (xb != PAD)] = UNK
                opt.zero_grad()
                loss_fn(self.net_(xb), ytr[b]).backward()
                nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.net_(xva), yva).item()
            self.history_.append(val_loss)
            if val_loss < best - 1e-4:
                best, best_state, bad = val_loss, copy.deepcopy(self.net_.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        self.net_.eval()
        x = self._encode(self._tokens(list(X)))
        with torch.no_grad():
            p = torch.sigmoid(self.net_(x)).numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
