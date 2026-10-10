import argparse
import math
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

from src.config import LABEL2ID, LABELS, MODELS, SEED
from src.evaluate import evaluate_probs, load_split, save_predictions

ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def tokenize(tokenizer, texts, labels, max_len):
    enc = tokenizer(list(texts), truncation=True, max_length=max_len)
    return [{"input_ids": ids, "attention_mask": mask, "labels": y}
            for ids, mask, y in zip(enc["input_ids"], enc["attention_mask"], labels)]


def class_weights(labels, power):
    counts = np.bincount(labels, minlength=len(LABELS))
    weights = (len(labels) / (len(LABELS) * counts)) ** power
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def predict_logits(model, tokenizer, texts, max_len, batch_size=256):
    model.eval()
    collate = DataCollatorWithPadding(tokenizer)
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), len(LABELS)), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        idx = order[start:start + batch_size]
        enc = tokenizer([texts[i] for i in idx], truncation=True, max_length=max_len)
        batch = collate([{"input_ids": a, "attention_mask": m} for a, m in zip(enc["input_ids"], enc["attention_mask"])])
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[idx] = model(**batch).logits.float().cpu().numpy()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cointegrated/rubert-tiny2")
    parser.add_argument("--name", default="rubert_tiny2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--weight-power", type=float, default=0.5)
    parser.add_argument("--evals-per-epoch", type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train, val = load_split("train"), load_split("val")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    ).cuda()

    train_items = tokenize(tokenizer, train["text"], train["label_id"], args.max_len)
    loader = DataLoader(train_items, batch_size=args.batch, shuffle=True, collate_fn=DataCollatorWithPadding(tokenizer))
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights(train["label_id"].values, args.weight_power).cuda())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total), total)
    eval_every = math.ceil(len(loader) / args.evals_per_epoch)

    out_dir = MODELS / args.name
    history, best, step = [], -1.0, 0
    val_texts = val["text"].tolist()
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % eval_every == 0 or step == total:
                pred = predict_logits(model, tokenizer, val_texts, args.max_len).argmax(1)
                f1 = f1_score(val["label_id"], pred, average="macro")
                history.append({"step": step, "epoch": round(step / len(loader), 2), "loss": round(loss.item(), 4),
                                "val_macro_f1": round(float(f1), 4), "minutes": round((time.time() - start) / 60, 1)})
                print(history[-1], flush=True)
                if f1 > best:
                    best = f1
                    model.save_pretrained(out_dir)
                    tokenizer.save_pretrained(out_dir)
                model.train()

    model = AutoModelForSequenceClassification.from_pretrained(out_dir).cuda()

    def predict_proba(texts):
        return torch.softmax(torch.from_numpy(predict_logits(model, tokenizer, list(texts), args.max_len)), 1).numpy()

    probs = save_predictions(args.name, predict_proba)
    evaluate_probs(args.name, probs, args.name, {
        "params": vars(args), "history": history, "best_val_macro_f1": round(best, 4),
        "train_minutes": round((time.time() - start) / 60, 1),
    })


if __name__ == "__main__":
    main()
