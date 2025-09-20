# code/evaluator_series/eval.py
import os, glob, argparse, time, json
from pathlib import Path

import pandas as pd
import torch

from evaluators.qwen import Qwen_Evaluator

choices = ["A", "B", "C", "D"]

# 指向 ceval-exam 根目录，如：~/offline_bundle/ceval/ceval-exam
DATA_ROOT = os.path.expanduser(os.environ.get("CEVAL_DATA_DIR", "data"))

def resolve_split_path(root: str, split: str, subject: str):
    """返回该 split 的数据路径（CSV 或 Parquet；可能是 list[str]）。"""
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
              f"Hint: Parquet 需要 pyarrow:  python -m pip install pyarrow")
        return None, None

def subject_display_name(key: str) -> str:
    # 优先读仓库根的 subject_mapping.json
    root = Path(__file__).resolve().parents[2]
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
    if "qwen" not in args.model_name.lower():
        print(f"Unknown model name (expected qwen family): {args.model_name}")
        return -1

    if args.cuda_device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluator = Qwen_Evaluator(
        choices=choices,
        k=args.ntrain,
        model_name=args.model_name,
        device=device,
        dtype=args.dtype
    )
    # 让评测器用你指定的判别模式（few-shot 也走判别，不再用生成+抽取）
    evaluator.score_mode = args.score_mode  # 'loglik_full' 或 'logits_first'

    subject_name = args.subject
    disp = subject_display_name(subject_name)

    os.makedirs("logs", exist_ok=True)
    run_date = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    tag = f"{Path(args.model_name).name}_{args.split}_{'fs'+str(args.ntrain) if args.few_shot else 'zs'}_{args.score_mode}"
    save_result_dir = os.path.join("logs", tag + "_" + run_date)
    os.makedirs(save_result_dir, exist_ok=True)

    print(subject_name)

    # 评测 split（支持 val/test）
    test_df, _ = load_split_df(DATA_ROOT, args.split, subject_name)
    assert test_df is not None, (
        f"找不到 {subject_name} 的 {args.split} 数据。\n"
        f"请检查 CEVAL_DATA_DIR={DATA_ROOT}。"
    )

    # few-shot 示例来源（默认 dev，更规范；如需从 val 取可改成 'val'）
    dev_df = None
    if args.few_shot:
        demo_split = args.demos_from  # 'dev' or 'val'
        dev_df, _ = load_split_df(DATA_ROOT, demo_split, subject_name)
        assert dev_df is not None, f"few_shot=True 但未找到 {subject_name} 的 {demo_split} 数据。"
        print(f"[INFO] few-shot demos from {demo_split}: k={args.ntrain}, available={len(dev_df)}")

    # 评测（评测器内部会把 header 加到 zero-shot / few-shot）
    correct_ratio = evaluator.eval_subject(
        subject_name=subject_name,
        test_df=test_df,
        dev_df=dev_df,
        few_shot=args.few_shot,
        cot=args.cot,
        save_result_dir=save_result_dir,
        subject_title=disp,  # 展示/提示词用标题
    )
    print("Acc:", correct_ratio)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--subject", "-s", type=str, default="operating_system")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--demos_from", choices=["dev", "val"], default="dev",
                        help="few-shot 示例来源，规范推荐 dev")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument(
        "--score_mode",
        choices=["logits_first", "loglik_full", "generate"],
        default="logits_first"
    )
    parser.add_argument("--dtype", choices=["fp16","bf16","fp32"], default="fp16")
    parser.add_argument("--cuda_device", type=str)
    args = parser.parse_args()
    main(args)
