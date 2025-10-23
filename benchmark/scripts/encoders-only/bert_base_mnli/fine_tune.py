#!/usr/bin/env python
import os
import json
import re
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)

OUT_DIR = "/content/outputs"
TRAIN_IO_FILES = ["/content/compliance_train.json"]
TEST_SEGMENTS_FILES = ["/content/raw_eval.json"]
KEYPOINTS_FILES = ["/content/GDPR_articles_obligations_keypoints_organized.json"]
METRICS_JSON_PATH = "/content/outputs/metrics.json"

MODEL_NAME = "ishan/bert-base-uncased-mnli"
LABELS = ["COMPLIANT", "NON_COMPLIANT", "NOT_APPLICABLE"]
MAX_SEQ_LEN = 512
HYP_TEMPLATE = 'The organization satisfies this key point: "{keypoint}".'
EPOCHS = 6
PER_DEVICE_TRAIN_BATCH = 512
GRAD_ACCUM = 1
PER_DEVICE_EVAL_BATCH = 512
LR = 2e-5
GRADIENT_CHECKPOINTING = True


def clip(s, n=220):
    s = re.sub(r"\s+", " ", str(s).strip())
    return s if len(s) <= n else s[:n] + "…"


def parse_index_list(s: Any) -> set:
    if s is None:
        return set()
    s = str(s).strip()
    if not s:
        return set()
    return {int(x.strip()) for x in s.split(",") if x.strip().isdigit()}


def macro_metrics(y_true, y_pred, labels=LABELS) -> Dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return {"Macro-P": float(p), "Macro-R": float(r), "Macro-F1": float(f1), "Accuracy": float(acc)}


def extract_excerpt_and_obligation(instr: str) -> Dict[str, str]:
    quotes = re.findall(r'"([^"]+)"', instr, flags=re.DOTALL)
    seg, kp = None, None
    if len(quotes) >= 2:
        seg, kp = quotes[0], quotes[1]
    else:
        parts = instr.split("Obligation:")
        if len(parts) >= 2:
            q_after = re.findall(r'"([^"]+)"', parts[1], flags=re.DOTALL)
            kp = q_after[0] if q_after else parts[1].strip()
            q_before = re.findall(r'"([^"]+)"', parts[0], flags=re.DOTALL)
            seg = q_before[0] if q_before else parts[0].strip()
    if not seg or not kp:
        raise ValueError("Could not parse 'excerpt' and 'obligation' from instruction.")
    return {"segment": seg.strip(), "keypoint_text": kp.strip()}


def load_io_json(files: List[str]) -> pd.DataFrame:
    rows = []
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for obj in data:
            instr = obj.get("instruction")
            out = obj.get("output")
            if not instr or not out:
                continue
            out = str(out).strip().upper()
            if out not in {"COMPLIANT", "NON_COMPLIANT", "NOT_APPLICABLE"}:
                continue
            try:
                ex = extract_excerpt_and_obligation(instr)
            except Exception:
                continue
            rows.append({"segment": ex["segment"], "keypoint_text": ex["keypoint_text"], "gold_label": out})
    if not rows:
        raise ValueError("No valid instruction/output examples found.")
    df = pd.DataFrame(rows)
    df["segment_id"] = np.arange(1, len(df) + 1)
    return df


def load_segments_flat(files: List[str]) -> pd.DataFrame:
    rows = []
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for s in data:
            if not all(k in s for k in ["gdpr_article_number", "label", "segment"]):
                continue
            rows.append(
                {
                    "gdpr_article_number": int(s["gdpr_article_number"]),
                    "segment": str(s["segment"]),
                    "label": str(s["label"]).strip().upper(),
                    "met_points": s.get("met_points", ""),
                    "unmet_points": s.get("unmet_points", ""),
                    "policy_id": s.get("policy_id"),
                    "batch_id": s.get("batch_id"),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No segments found.")
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)
    df["segment_id"] = np.arange(1, len(df) + 1)
    return df


def load_keypoints(files: List[str]) -> Dict[int, List[str]]:
    kp_map: Dict[int, List[str]] = {}
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            continue
        for obj in data:
            if isinstance(obj, dict) and "gdpr_article" in obj and "keypoints" in obj:
                art = int(obj["gdpr_article"])
                kp_map[art] = [str(x).strip() for x in obj["keypoints"]]
    if not kp_map:
        raise ValueError("No keypoints found.")
    return kp_map


def expand_pairs_from_eval(segments_df: pd.DataFrame, kp_map: Dict[int, List[str]]) -> pd.DataFrame:
    rows = []
    for _, r in segments_df.iterrows():
        art = int(r["gdpr_article_number"])
        kps = kp_map.get(art)
        if not kps:
            continue
        met, unmet = parse_index_list(r["met_points"]), parse_index_list(r["unmet_points"])
        K = len(kps)
        for idx in range(1, K + 1):
            if idx in met:
                gold = "COMPLIANT"
            elif idx in unmet:
                gold = "NON_COMPLIANT"
            else:
                gold = "NOT_APPLICABLE"
            rows.append(
                {
                    "segment_id": int(r["segment_id"]),
                    "policy_id": r.get("policy_id"),
                    "gdpr_article_number": art,
                    "keypoint_index": idx,
                    "keypoint_text": kps[idx - 1],
                    "segment": r["segment"],
                    "gold_label": gold,
                }
            )
    return pd.DataFrame(rows)


def encode_pair(tokenizer, segment: str, keypoint_text: str, gold_label: str, task2idx: Dict[str, int]):
    pair = HYP_TEMPLATE.format(keypoint=keypoint_text)
    enc = tokenizer(segment, pair, truncation=True, max_length=MAX_SEQ_LEN)
    enc["labels"] = task2idx[gold_label]
    return enc


class PairsDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, task2idx: Dict[str, int]):
        self.df = df.reset_index(drop=True)
        self.enc = [
            encode_pair(tokenizer, self.df.iloc[i]["segment"], self.df.iloc[i]["keypoint_text"], self.df.iloc[i]["gold_label"], task2idx)
            for i in range(len(self.df))
        ]

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.enc[idx].items()}


def compute_metrics_fn_factory(idx2task: Dict[int, str]):
    def compute_metrics_fn(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        y_true = [idx2task[int(x)] for x in labels]
        y_pred = [idx2task[int(x)] for x in preds]
        m = macro_metrics(y_true, y_pred, labels=LABELS)
        return {"macro_p": m["Macro-P"], "macro_r": m["Macro-R"], "macro_f1": m["Macro-F1"], "accuracy": m["Accuracy"]}
    return compute_metrics_fn


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    train_pairs_df = load_io_json(TRAIN_IO_FILES)
    test_segments_df = load_segments_flat(TEST_SEGMENTS_FILES)
    keypoints_map = load_keypoints(KEYPOINTS_FILES)
    test_pairs_df = expand_pairs_from_eval(test_segments_df, keypoints_map)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    id2label_cfg = {int(k): str(v).upper() for k, v in model.config.id2label.items()}
    label2id_cfg = {v: k for k, v in id2label_cfg.items()}
    task2idx = {
        "COMPLIANT": label2id_cfg["ENTAILMENT"],
        "NON_COMPLIANT": label2id_cfg["CONTRADICTION"],
        "NOT_APPLICABLE": label2id_cfg["NEUTRAL"],
    }
    idx2task = {v: k for k, v in task2idx.items()}

    if GRADIENT_CHECKPOINTING:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    train_dataset = PairsDataset(train_pairs_df, tokenizer, task2idx)
    eval_dataset = PairsDataset(test_pairs_df, tokenizer, task2idx)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    args = TrainingArguments(
        output_dir=OUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        save_strategy="no",
        logging_strategy="no",
        evaluation_strategy="no",
        report_to=[],
    )

    class BnbPagedAdamW8bitTrainer(Trainer):
        def create_optimizer(self):
            if self.optimizer is None:
                import bitsandbytes as bnb
                params = [p for p in self.model.parameters() if p.requires_grad]
                self.optimizer = bnb.optim.PagedAdamW8bit(params, lr=LR, weight_decay=0.0)
            return self.optimizer

    trainer = BnbPagedAdamW8bitTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_fn_factory(idx2task),
    )

    trainer.train()

    pred_out = trainer.predict(eval_dataset)
    test_logits = pred_out.predictions
    test_preds_ids = test_logits.argmax(-1)
    y_true_ids = pred_out.label_ids

    y_true = [idx2task[int(x)] for x in y_true_ids]
    y_pred = [idx2task[int(x)] for x in test_preds_ids]
    m1 = macro_metrics(y_true, y_pred, labels=LABELS)

    scored = test_pairs_df.copy()
    scored["pred_label"] = y_pred

    if scored.get("policy_id", pd.Series(dtype=object)).notna().any():
        group_keys = ["policy_id", "gdpr_article_number"]
    else:
        group_keys = ["segment_id", "gdpr_article_number"]

    agg_rows = []
    for _, grp in scored.groupby(group_keys):
        labels_grp = grp["pred_label"].tolist()
        true_grp = grp["gold_label"].tolist()
        def aggregate_labels(labels):
            if all(l == "NOT_APPLICABLE" for l in labels):
                return "NOT_APPLICABLE"
            share_c = sum(l == "COMPLIANT" for l in labels) / len(labels)
            return "COMPLIANT" if share_c > 0.5 else "NON_COMPLIANT"
        gold = aggregate_labels(true_grp)
        pred = aggregate_labels(labels_grp)
        agg_rows.append({"article": int(grp["gdpr_article_number"].iloc[0]), "gold_article_label": gold, "pred_article_label": pred})
    agg_df = pd.DataFrame(agg_rows)

    m2 = macro_metrics(agg_df["gold_article_label"], agg_df["pred_article_label"], labels=LABELS)

    def per_article_f1(article_ids, y_t, y_p) -> pd.DataFrame:
        df_tmp = pd.DataFrame({"article": article_ids, "y_true": y_t, "y_pred": y_p})
        rows = []
        for a, grp in df_tmp.groupby("article"):
            p, r, f1, _ = precision_recall_fscore_support(grp["y_true"], grp["y_pred"], labels=LABELS, average="macro", zero_division=0)
            rows.append({"article": int(a), "macro_p": float(p), "macro_r": float(r), "macro_f1": float(f1), "support": int(len(grp))})
        return pd.DataFrame(rows).sort_values("article")

    scores_by_article = per_article_f1(
        article_ids=agg_df["article"].tolist(),
        y_t=agg_df["gold_article_label"].tolist(),
        y_p=agg_df["pred_article_label"].tolist(),
    )
    macro_over_articles = float("nan") if scores_by_article.empty else float(scores_by_article["macro_f1"].mean())

    metrics_json = {
        "stage1": {"metrics": m1},
        "stage2": {"metrics": m2},
        "stage3": {
            "macro_f1_across_articles": float(macro_over_articles),
            "per_article": scores_by_article.to_dict(orient="records"),
        },
    }

    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
