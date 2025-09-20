# code/evaluator_series/eval.py
import os, glob, argparse, time, json
from pathlib import Path

import pandas as pd
import torch

from evaluators.qwen import Qwen_Evaluator

choices = ["A", "B", "C", "D"]

# 数据根（指向 ceval-exam 那一层）
DATA_ROOT = os.path.expanduser(os.environ.get("CEVAL_DATA_DIR", "data"))

def resolve_split_path(root: str, split: str, subject: str):
    """
    返回该 split 的数据路径（CSV 或 Parquet）。可能返回 str 或 list[str]（Parquet 分片）。
    支持：
      A) 聚合 CSV:        root/val/<sub>_val.csv
      B) 学科 CSV:        root/<sub>/val.csv
      C) 学科 Parquet:    root/<sub>/val-*.parquet
      D) 聚合 Parquet:    root/val/<sub>_val-*.parquet
    """
    root = os.path.expanduser(root)
    a = os.path.join(root, split, f"{subject}_{split}.csv")
    b = os.path.join(root, subject, f"{split}.csv")
    if os.path.exists(a): return a
    if os.path.exists(b): return b
    pq1 = sorted(glob.glob(os.path.join(root, subject, f"{split}-*.parquet")))
    if pq1: return pq1
    pq2 = sorted(glob.glob(os.path.join(root, split, f"{subject}_{split}-*.parquet")))
    if pq2: return pq2
    return None

def load_split_df(root: str, split: str, subject: str):
    path = resolve_split_path(root, split, subject)
    if path is None:
        return None, None
    print(f"[INFO] Using {split} data for '{subject}': {path}")
    try:
        if isinstance(path, list):  # parquet 分片
            dfs = [pd.read_parquet(p) for p in path]
            return pd.concat(dfs, ignore_index=True), path
        if str(path).endswith(".parquet"):
            return pd.read_parquet(path), path
        return pd.read_csv(path), path
    except Exception as e:
        print(f"[ERROR] Failed to read {split} for {subject}: {e}\n"
              f"Hint: Parquet 需要安装 pyarrow： pip install pyarrow")
        return None, None

def subject_display_name(key: str) -> str:
    root = Path(__file__).resolve().parents[2]  # 仓库根
    mp = root / "subject_mapping.json"
    if mp.exists():
        try:
            mpd = json.loads(mp.read_text(encoding="utf-8"))
            v = mpd.get(key)
            if isinstance(v, str):
                return v
            if isinstance(v, dict):
                return v.get("zh") or v.get("cn") or v.get("en") or key
        except Exception:
            pass
    return key.replace("_", " ").title()

def main(args):
    name_l = args.model_name.lower()
    if "qwen" not in name_l:
        print(f"Unknown model name (expected qwen family): {args.model_name}")
        return -1

    if args.cuda_device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluator = Qwen_Evaluator(
        choices=choices,
        k=args.ntrain,
        model_name=args.model_name,
        device=device
    )

    evaluator.score_mode = args.score_mode  # logits_first / loglik_full

    subject_name = args.subject
    disp = subject_display_name(subject_name) 

    os.makedirs("logs", exist_ok=True)
    run_date = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    save_result_dir = os.path.join("logs", f"{Path(args.model_name).name}_{run_date}")
    os.makedirs(save_result_dir, exist_ok=True)

    print(subject_name)

    # 读 val（
    val_df, _ = load_split_df(DATA_ROOT, "val", subject_name)
    assert val_df is not None, (
        f"找不到 {subject_name} 的 val 数据（CSV/Parquet）。\n"
        f"请检查 CEVAL_DATA_DIR={DATA_ROOT} 是否指向 ceval-exam 根目录。"
    )

    # few-shot 
    dev_df = None
    if args.few_shot:
        dev_df, _ = load_split_df(DATA_ROOT, "dev", subject_name)
        assert dev_df is not None, f"few_shot=True 但未找到 {subject_name} 的 dev 数据。"
        print(f"[INFO] few-shot demos from dev: k={args.ntrain}, available={len(dev_df)}")

    correct_ratio = evaluator.eval_subject(
        subject_name=subject_name,
        test_df=val_df,
        dev_df=dev_df,
        few_shot=args.few_shot,
        cot=args.cot,
        save_result_dir=save_result_dir,
        subject_title=disp,
    )
    print("Acc:", correct_ratio)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--subject", "-s", type=str, default="operating_system")
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--score_mode",
                        choices=["logits_first", "loglik_full"],
                        default="logits_first")
    parser.add_argument("--cuda_device", type=str)
    args = parser.parse_args()
    main(args)
