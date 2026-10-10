import argparse
import json
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.config import LABELS, MODELS, SEED
from src.evaluate import METRICS, evaluate_probs, load_split, save_predictions

NAVEC = MODELS / "navec" / "navec_hudlit_v1_12B_500K_300d_100q.tar"
PAD, UNK = 0, 1


class Vocab:
    def __init__(self, texts, min_freq=2, max_size=150_000):
        counts = Counter(word for text in texts for word in text.split())
        words = [w for w, c in counts.most_common(max_size) if c >= min_freq]
        self.words = ["<pad>", "<unk>"] + words
        self.index = {w: i for i, w in enumerate(self.words)}

    def encode(self, text, max_len):
        ids = [self.index.get(w, UNK) for w in text.split()[:max_len]]
        return ids or [UNK]

    def embeddings(self, dim=300):
        from navec import Navec
        navec = Navec.load(NAVEC)
        rng = np.random.default_rng(SEED)
        known = np.stack([navec[w] for w in self.words[2:] if w in navec])
        matrix = rng.normal(0, known.std(), (len(self.words), dim)).astype(np.float32)
        matrix[PAD] = 0
        hits = 0
        for i, word in enumerate(self.words[2:], start=2):
            if word in navec:
                matrix[i] = navec[word]
                hits += 1
        return torch.from_numpy(matrix), hits / (len(self.words) - 2)


class BiGRU(nn.Module):
    def __init__(self, embeddings, hidden=192, layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=PAD)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(embeddings.shape[1], hidden, layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Linear(4 * hidden, len(LABELS))

    def forward(self, ids):
        mask = (ids != PAD).unsqueeze(-1)
        out, _ = self.rnn(self.dropout(self.embedding(ids)))
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        maxed = out.masked_fill(~mask, -1e4).max(1).values
        return self.head(self.dropout(torch.cat([mean, maxed], 1)))


class TextCNN(nn.Module):
    def __init__(self, embeddings, kernels=(1, 2, 3, 4, 5), filters=128, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=PAD)
        self.convs = nn.ModuleList(nn.Conv1d(embeddings.shape[1], filters, k, padding=k // 2) for k in kernels)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(filters * len(kernels), len(LABELS))

    def forward(self, ids):
        mask = (ids != PAD).unsqueeze(1)
        x = self.dropout(self.embedding(ids)).transpose(1, 2)
        pooled = [F.relu(conv(x))[:, :, :ids.shape[1]].masked_fill(~mask, -1e4).max(2).values for conv in self.convs]
        return self.head(self.dropout(torch.cat(pooled, 1)))


ARCHITECTURES = {"bigru": BiGRU, "textcnn": TextCNN}


def collate(batch):
    ids = [torch.tensor(x) for x, _ in batch]
    labels = torch.tensor([y for _, y in batch])
    return nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=PAD), labels


@torch.no_grad()
def predict_proba(model, vocab, texts, max_len, batch_size=1024):
    device = next(model.parameters()).device
    model.eval()
    encoded = [vocab.encode(t, max_len) for t in texts]
    order = np.argsort([len(x) for x in encoded])
    out = np.zeros((len(texts), len(LABELS)), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        idx = order[start:start + batch_size]
        ids, _ = collate([(encoded[i], 0) for i in idx])
        out[idx] = torch.softmax(model(ids.to(device)).float(), 1).cpu().numpy()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=list(ARCHITECTURES), required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=100)
    parser.add_argument("--weight-power", type=float, default=0.5)
    args = parser.parse_args()
    name = f"{args.arch}_navec"

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train, val = load_split("train"), load_split("val")
    vocab = Vocab(train["text"])
    embeddings, coverage = vocab.embeddings()
    print(f"словарь: {len(vocab.words)}, покрытие navec: {coverage:.1%}", flush=True)

    model = ARCHITECTURES[args.arch](embeddings).cuda()
    items = [(vocab.encode(t, args.max_len), y) for t, y in zip(train["text"], train["label_id"])]
    loader = DataLoader(items, batch_size=args.batch, shuffle=True, collate_fn=collate)
    counts = np.bincount(train["label_id"], minlength=len(LABELS))
    weights = torch.tensor((len(train) / (len(LABELS) * counts)) ** args.weight_power, dtype=torch.float32).cuda()
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=len(loader) * args.epochs, pct_start=0.1)

    path = MODELS / f"{name}.pt"
    history, best, start = [], -1.0, time.time()
    val_texts = val["text"].tolist()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for ids, labels in loader:
            loss = loss_fn(model(ids.cuda()), labels.cuda())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        f1 = f1_score(val["label_id"], predict_proba(model, vocab, val_texts, args.max_len).argmax(1), average="macro")
        history.append({"epoch": epoch, "loss": round(loss.item(), 4), "val_macro_f1": round(float(f1), 4),
                        "minutes": round((time.time() - start) / 60, 1)})
        print(history[-1], flush=True)
        if f1 > best:
            best = f1
            torch.save(model.state_dict(), path)

    model.load_state_dict(torch.load(path))
    probs = save_predictions(name, lambda texts: predict_proba(model, vocab, texts, args.max_len))
    evaluate_probs(name, probs, f"{args.arch.upper()} + navec", {
        "params": vars(args), "history": history, "vocab_size": len(vocab.words),
        "navec_coverage": round(coverage, 4), "train_minutes": round((time.time() - start) / 60, 1),
        "parameters": sum(p.numel() for p in model.parameters()),
    })


if __name__ == "__main__":
    main()
