# batch_eval_inprocess.py
import os, glob, argparse, json
import pandas as pd
import torch
from evaluators.qwen import Qwen_Evaluator

CHOICES = ["A","B","C","D"]

def key_to_title(key: str) -> str:
    if key.startswith("high_school_"):
        return "High School " + key[len("high_school_"):].replace("_"," ").title()
    if key.startswith("middle_school_"):
        return "Middle School " + key[len("middle_school_"):].replace("_"," ").title()
    return key.replace("_"," ").title()

def detect_group(key: str) -> str:
    if key.startswith("high_school_"): return "High School"
    if key.startswith("middle_school_"): return "Middle School"
    return "Other"

def resolve_split_path(root: str, split: str, sub: str):
    a = os.path.join(root, split, f"{sub}_{split}.csv")  # 聚合式 CSV
    b = os.path.join(root, sub, f"{split}.csv")          # 学科式 CSV
    if os.path.exists(a): return a
    if os.path.exists(b): return b
    pq1 = sorted(glob.glob(os.path.join(root, sub, f"{split}-*.parquet")))   # 学科式 Parquet
    if pq1: return pq1
    pq2 = sorted(glob.glob(os.path.join(root, split, f"{sub}_{split}-*.parquet")))  # 聚合式 Parquet（少见）
    if pq2: return pq2
    return None

def load_split_df(root: str, split: str, sub: str):
    p = resolve_split_path(root, split, sub)
    if p is None: return None, None
    if isinstance(p, list):  # parquet sharded
        dfs = [pd.read_parquet(x) for x in p]
        return pd.concat(dfs, ignore_index=True), p
    return pd.read_csv(p), p

def discover_subjects(root: str, restrict_group: str|None):
    subs = []
    # 学科式目录
    for d in os.listdir(root):
        p = os.path.join(root, d)
        if not os.path.isdir(p): continue
        if restrict_group == "High School" and not d.startswith("high_school_"): continue
        if restrict_group == "Middle School" and not d.startswith("middle_school_"): continue
        if resolve_split_path(root, "val", d): subs.append(d)
    # 聚合式目录（补充）
    vdir = os.path.join(root, "val")
    if os.path.isdir(vdir):
        for f in os.listdir(vdir):
            if f.endswith("_val.csv"):
                subs.append(f[:-8])  # 去掉 _val.csv
            elif "_val-" in f and f.endswith(".parquet"):
                subs.append(f.split("_val-")[0])
    return sorted(set(subs))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True, help="本地模型目录或HF模型名")
    ap.add_argument("--data_root", required=True, help="指向 ceval-exam 根目录")
    ap.add_argument("--split", default="val", choices=["dev","val","test"])
    ap.add_argument("--group", choices=["High School","Middle School","Both"], default="Both")
    ap.add_argument("--subjects", nargs="*", default=None, help="可显式指定学科key（如 high_school_biology）")
    ap.add_argument("--out_csv", default="./outputs/summary_val.csv")
    ap.add_argument("--save_each_dir", default="./outputs/per_subject", help="逐题输出目录")
    ap.add_argument("--dtype", choices=["fp16","bf16","fp32"], default="fp16")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(args.save_each_dir, exist_ok=True)

    # 选科目
    if args.subjects:
        subjects = args.subjects
    else:
        if args.group == "Both":
            subjects = discover_subjects(args.data_root, None)
            subjects = [s for s in subjects if s.startswith("high_school_") or s.startswith("middle_school_")]
        else:
            subjects = discover_subjects(args.data_root, args.group)
    assert subjects, f"No subjects found under {args.data_root}"

    # 模型 & 设备
    if args.dtype=="fp16": dtype=torch.float16
    elif args.dtype=="bf16": dtype=torch.bfloat16
    else: dtype=torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 用已有的评测器（一次加载模型，循环评）
    evaluator = Qwen_Evaluator(choices=CHOICES, k=5, model_name=args.model_name_or_path, device=torch.device(device), dtype=args.dtype)

    rows = []
    for sub in subjects:
        df, src = load_split_df(args.data_root, args.split, sub)
        if df is None:
            print(f"[WARN] skip {sub}: no {args.split} data")
            continue
        n = len(df)
        title = key_to_title(sub)
        group = detect_group(sub)
        print(f"\n=== {title} ({sub}) | {args.split}={n} ===")
        acc_pct = evaluator.eval_subject(sub, df, dev_df=None, few_shot=False,
                                         save_result_dir=args.save_each_dir, cot=False)
        # 兼容返回比例/百分比
        acc = float(acc_pct)
        if acc > 1.0: acc = acc / 100.0
        rows.append({
            "subject_key": sub,
            "subject_title": title,
            "group": group,
            "split": args.split,
            "n": n,
            "acc": acc
        })

    out_df = pd.DataFrame(rows).sort_values(["group","subject_key"])
    out_df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"\nSaved summary: {args.out_csv}")
    if not out_df.empty:
        overall = out_df["acc"].mean()
        print(f"Overall average accuracy: {overall:.4f}")

if __name__ == "__main__":
    main()
