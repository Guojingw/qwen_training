import os
import glob
import argparse
import pandas as pd
import torch
import time

from evaluators.qwen import Qwen_Evaluator

choices = ["A", "B", "C", "D"]

# 数据根：默认 ./data，可用环境变量覆盖（指向 ceval-exam 那一层）
DATA_ROOT = os.environ.get("CEVAL_DATA_DIR", "data")

def resolve_split_path(root: str, split: str, subject: str):
    """
    返回该 split 的数据路径（CSV 或 Parquet）。可能返回 str 或 list[str]（Parquet 分片）。
    支持：
      A) 聚合式 CSV:   root/val/<sub>_val.csv
      B) 学科式 CSV:   root/<sub>/val.csv
      C) 学科式 Parquet: root/<sub>/val-*.parquet
      D) 聚合式 Parquet: root/val/<sub>_val-*.parquet
    """
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
    # 打印一下用到的文件，便于排错
    print(f"[INFO] Using {split} data for '{subject}': {path}")
    if isinstance(path, list):  # parquet 分片
        dfs = [pd.read_parquet(p) for p in path]
        return pd.concat(dfs, ignore_index=True), path
    # CSV
    return pd.read_csv(path), path

def main(args):
    # 仅 Qwen 分支
    if "qwen" in args.model_name.lower() or "Qwen3" in args.model_name:
        if args.cuda_device:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        evaluator = Qwen_Evaluator(
            choices=choices,
            k=args.ntrain,
            model_name=args.model_name,
            device=device,
            score_mode = args.score_mode
        )
    else:
        print("Unknown model name (expected qwen family)")
        return -1

    subject_name = args.subject
    os.makedirs("logs", exist_ok=True)
    run_date = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    save_result_dir = os.path.join("logs", f"{args.model_name}_{run_date}")
    os.makedirs(save_result_dir, exist_ok=True)

    print(subject_name)

    # 读取 val（CSV/Parquet 皆可；两种目录结构皆可）
    val_df, val_src = load_split_df(DATA_ROOT, "val", subject_name)
    if val_df is None:
        raise AssertionError(f"找不到 {subject_name} 的 val 数据（CSV/Parquet）。"
                             f"\n请检查 CEVAL_DATA_DIR={DATA_ROOT} 是否指向 ceval-exam 根目录，"
                             f"以及目录名是否为真实学科名。")

    # few-shot 需要 dev
    dev_df = None
    if args.few_shot:
        dev_df, dev_src = load_split_df(DATA_ROOT, "dev", subject_name)
        if dev_df is None:
            raise AssertionError(f"few_shot=True 但未找到 {subject_name} 的 dev 数据（CSV/Parquet）。")

    # 评测
    correct_ratio = evaluator.eval_subject(
        subject_name, 
        val_df,
        dev_df=dev_df,
        few_shot=args.few_shot,
        save_result_dir=save_result_dir,
        cot=args.cot,
        score_mode = args.score_mode
    )
    print("Acc:", correct_ratio)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--subject", "-s", type=str, default="operating_system")
    parser.add_argument("--cuda_device", type=str)
    parser.add_argument("--score_mode", choices=["logits_first","loglik_full"], default="logits_first")
    args = parser.parse_args()
    
    main(args)
