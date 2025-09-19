import os
import argparse
import pandas as pd
import torch

from evaluators.qwen import Qwen_Evaluator

import time
choices = ["A", "B", "C", "D"]

# 允许用环境变量指定数据根目录；未设置则默认使用 ./data
DATA_ROOT = os.environ.get("CEVAL_DATA_DIR", "data")

def resolve_csv(root, split, subject):
    """兼容两种布局：
       A) root/val/<sub>_val.csv
       B) root/<sub>/val.csv
    """
    a = os.path.join(root, split, f"{subject}_{split}.csv")
    b = os.path.join(root, subject, f"{split}.csv")
    if os.path.exists(a): return a
    if os.path.exists(b): return b
    return None

def main(args):

    if "qwen" in args.model_name.lower():
        if args.cuda_device:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        evaluator = Qwen_Evaluator(
            choices=choices,
            k=args.ntrain,
            model_name=args.model_name,
            device=device
        )
    else:
        print("Unknown model name")
        return -1

    subject_name=args.subject
    if not os.path.exists(r"logs"):
        os.mkdir(r"logs")
    run_date=time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
    save_result_dir=os.path.join(r"logs",f"{args.model_name}_{run_date}")
    os.mkdir(save_result_dir)
    print(subject_name)
    val_file_path = resolve_csv(DATA_ROOT, "val", subject_name)
    assert val_file_path, f"找不到 {subject_name} 的 val CSV，请检查 CEVAL_DATA_DIR 或数据目录结构"
    val_df = pd.read_csv(val_file_path)

    dev_df = None
    if args.few_shot:
        dev_file_path = resolve_csv(DATA_ROOT, "dev", subject_name)
        assert dev_file_path, f"few_shot 需要 dev CSV，但未找到 {subject_name} 的 dev 数据"
        dev_df = pd.read_csv(dev_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--openai_key", type=str,default="xxx")
    parser.add_argument("--minimax_group_id", type=str,default="xxx")
    parser.add_argument("--minimax_key", type=str,default="xxx")
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--model_name",type=str)
    parser.add_argument("--cot",action="store_true")
    parser.add_argument("--subject","-s",type=str,default="operating_system")
    parser.add_argument("--cuda_device", type=str)    
    args = parser.parse_args()
    main(args)