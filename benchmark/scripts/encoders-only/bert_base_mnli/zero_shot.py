#!/usr/bin/env python
import json
import re
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

OUT_DIR = "/content/outputs"
TEST_SEGMENTS_FILES = ["/content/raw_eval.json"]
KEYPOINTS_FILES = ["/content/GDPR_articles_obligations_keypoints_organized.json"]
METRICS_JSON_PATH = "/content/outputs/metrics.json"

LABELS = ["COMPLIANT", "NON_COMPLIANT", "NOT_APPLICABLE"]
NLI_MODEL_NAME = "ishan/bert-base-uncased-mnli"
MAX_SEQ_LEN = 512
NLI_BATCH = 32

def make_hypothesis(keypoint_text: str) -> str:
    return f'The organization satisfies this key point: "{keypoint_text}".'

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
                    "justification": s.get("justification"),
                    "compliance_score": s.get("compliance_score"),
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

def expand_pairs(segments_df: pd.DataFrame, kp_map: Dict[int, List[str]]) -> pd.DataFrame:
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

def nli_predict_batch(model, tokenizer, device, premises: List[str], hypotheses: List[str]) -> List[str]:
    assert len(premises) == len(hypotheses)
    preds = []
    with torch.no_grad():
        for i in range(0, len(premises), NLI_BATCH):
            batch_p = premises[i : i + NLI_BATCH]
            batch_h = hypotheses[i : i + NLI_BATCH]
            enc = tokenizer(batch_p, batch_h, truncation=True, max_length=MAX_SEQ_LEN, padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits
            idx = logits.argmax(dim=-1).detach().cpu().tolist()
            preds.extend([map_nli_to_task(model.config.id2label[j]) for j in idx])
    return preds

def map_nli_to_task(nli_label: str) -> str:
    nli_label = str(nli_label).upper()
    if "ENTAIL" in nli_label:
        return "COMPLIANT"
    if "CONTRADICT" in nli_label:
        return "NON_COMPLIANT"
    return "NOT_APPLICABLE"

def aggregate_labels(labels: List[str]) -> str:
    if all(l == "NOT_APPLICABLE" for l in labels):
        return "NOT_APPLICABLE"
    share_c = sum(l == "COMPLIANT" for l in labels) / len(labels)
    return "COMPLIANT" if share_c > 0.5 else "NON_COMPLIANT"

def per_article_f1(article_ids, y_true, y_pred) -> pd.DataFrame:
    df_tmp = pd.DataFrame({"article": article_ids, "y_true": y_true, "y_pred": y_pred})
    rows = []
    for a, grp in df_tmp.groupby("article"):
        p, r, f1, _ = precision_recall_fscore_support(grp["y_true"], grp["y_pred"], labels=LABELS, average="macro", zero_division=0)
        rows.append({"article": int(a), "macro_p": float(p), "macro_r": float(r), "macro_f1": float(f1), "support": int(len(grp))})
    return pd.DataFrame(rows).sort_values("article")

def main() -> None:
    test_segments_df = load_segments_flat(TEST_SEGMENTS_FILES)
    keypoints_map = load_keypoints(KEYPOINTS_FILES)
    test_pairs_df = expand_pairs(test_segments_df, keypoints_map)

    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    all_premises = test_pairs_df["segment"].tolist()
    all_hypotheses = [make_hypothesis(kp) for kp in test_pairs_df["keypoint_text"].tolist()]
    y_pred = nli_predict_batch(model, tokenizer, device, all_premises, all_hypotheses)
    y_true = test_pairs_df["gold_label"].tolist()

    m1 = macro_metrics(y_true, y_pred)

    scored = test_pairs_df.copy()
    scored["pred_label"] = y_pred

    if scored["policy_id"].notna().any():
        group_keys = ["policy_id", "gdpr_article_number"]
    else:
        group_keys = ["segment_id", "gdpr_article_number"]

    agg_rows = []
    for _, grp in scored.groupby(group_keys):
        gold = aggregate_labels(grp["gold_label"].tolist())
        pred = aggregate_labels(grp["pred_label"].tolist())
        agg_rows.append(
            {
                "article": int(grp["gdpr_article_number"].iloc[0]),
                "gold_article_label": gold,
                "pred_article_label": pred,
            }
        )
    agg_df = pd.DataFrame(agg_rows)

    m2 = macro_metrics(agg_df["gold_article_label"], agg_df["pred_article_label"])

    scores_by_article = per_article_f1(
        article_ids=agg_df["article"].tolist(),
        y_true=agg_df["gold_article_label"].tolist(),
        y_pred=agg_df["pred_article_label"].tolist(),
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
