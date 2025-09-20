# mini_ceval_eval.py
import os, re, glob, argparse, random, math
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def header(subject):
    return (
        f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n"
        f"请只输出两行：\n"
        f"思路：一句话简要推理。\n"
        f"答案：仅写A/B/C/D。\n\n"
    )

def fmt_example(sample, is_demo):
    q = sample.get("question", "")
    opts = "\n".join(f"{k}. {sample.get(k,'')}" for k in ["A","B","C","D"])
    if is_demo:
        expl = (sample.get("explanation") or "略").strip()
        ans  = str(sample.get("answer","")).strip().upper()
        return f"题目：{q}\n{opts}\n思路：{expl}\n答案：{ans}\n\n"
    else:
        # 待预测
        return (
            f"题目：{q}\n{opts}\n"
            f"思路："   # 让模型先给一句“思路”，下一行给“答案”
        )

def extract_choice(text: str):
    # 先找“答案：X”
    for line in text.splitlines():
        if "答案" in line:
            m = re.search(r"[:：]\s*([ABCD])\b", line)
            if m: return m.group(1)
    # 找到最后一次出现的 A/B/C/D（兜底）
    m = list(re.finditer(r"\b([ABCD])\b", text))
    return m[-1].group(1) if m else None

def clean_and_extract(text: str):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    idea, ans = None, None
    for l in lines:
        m = re.search(r"思路[:：]\s*(.*)", l)
        if m: idea = m.group(1); break
    for l in lines:
        m = re.search(r"答案[:：]\s*([ABCD])\b", l)
        if m: ans = m.group(1); break
    if ans is None:
        ans = extract_choice("\n".join(lines))
    return f"思路：{idea or ''}\n答案：{ans or ''}", ans

def list_subjects(root: str, group: str):
    # 从本地 ceval-exam 目录列学科（含 high_school/middle_school）
    subs = []
    for p in sorted(Path(root).glob("*")):
        if not p.is_dir(): continue
        name = p.name
        if group == "all":
            if ("high_school" in name) or ("middle_school" in name):
                subs.append(name)
        elif group == "high_school":
            if "high_school" in name: subs.append(name)
        elif group == "middle_school":
            if "middle_school" in name: subs.append(name)
    return subs

def resolve_paths(root, subject, split):
    """返回 CSV 或 parquet list"""
    a = Path(root)/split/f"{subject}_{split}.csv"
    b = Path(root)/subject/f"{split}.csv"
    if a.exists(): return [str(a)]
    if b.exists(): return [str(b)]
    pq1 = sorted(glob.glob(str(Path(root)/subject/f"{split}-*.parquet")))
    if pq1: return pq1
    pq2 = sorted(glob.glob(str(Path(root)/split/f"{subject}_{split}-*.parquet")))
    if pq2: return pq2
    return []

def read_split(root, subject, split):
    paths = resolve_paths(root, subject, split)
    if not paths:
        return pd.DataFrame(), []
    dfs = []
    for p in paths:
        if p.endswith(".csv"):
            dfs.append(pd.read_csv(p))
        else:
            # 需要 pyarrow
            dfs.append(pd.read_parquet(p))
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    return df, paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="本地路径或HF模型名")
    ap.add_argument("--data_root", default=os.environ.get("CEVAL_DATA_DIR","./ceval/ceval-exam"))
    ap.add_argument("--split", choices=["val","test"], default="val")
    ap.add_argument("--group", choices=["high_school","middle_school","all"], default="high_school")
    ap.add_argument("--subjects_file", help="可选：逐行列出学科名，优先于 --group")
    ap.add_argument("-k","--k_shot", type=int, default=0, help="0=zero-shot；>0 表示从 val 采样 k 个示例")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--sampling", action="store_true", help="使用采样（top_p/top_k/temperature）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # 模型
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device).eval()

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        do_sample=args.sampling,
    )
    if args.sampling:
        gen_kwargs.update(dict(top_p=0.95, top_k=64, temperature=0.7))

    # 学科列表
    if args.subjects_file:
        with open(args.subjects_file, "r", encoding="utf-8") as f:
            subjects = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
    else:
        subjects = list_subjects(args.data_root, args.group)

    # 评测
    total_correct = total_cnt = 0
    print(f"=== split={args.split} | k-shot={args.k_shot} | model={args.model} ===")
    print(f"{'subject':28s} {'n':>4s} {'acc':>6s}")

    for sub in subjects:
        test_df, test_src = read_split(args.data_root, sub, args.split)
        if test_df.empty:
            print(f"{sub:28s} {'0':>4s} {'NA':>6s}  (missing {args.split})")
            continue

        # few-shot demos 来自 val；若 split=val 也 OK（示例和待测会重复，但这是你提供脚本中的做法）
        demos_df = None
        if args.k_shot > 0:
            demos_df, _ = read_split(args.data_root, sub, "val")
            if demos_df is None or len(demos_df) == 0:
                print(f"[WARN] {sub} 没有 val，回退 zero-shot")
                args.k_shot = 0

        samples = test_df.to_dict(orient="records")
        preds, golds = [], [str(x).strip().upper() for x in test_df["answer"].tolist()]

        # 组 batch
        for i in range(0, len(samples), args.batch):
            batch = samples[i:i+args.batch]

            # 构造本批 few-shot 前缀（与你参考代码一致：每批随机采样 k 个）
            prefix = ""
            if args.k_shot > 0:
                few = random.sample(demos_df.to_dict(orient="records"), k=min(args.k_shot, len(demos_df)))
                prefix = header(sub) + "".join(fmt_example(s, True) for s in few)
            else:
                prefix = header(sub)

            prompts = [prefix + fmt_example(s, False) for s in batch]

            with torch.no_grad():
                inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
                outputs = mdl.generate(**inputs, **gen_kwargs)

            # 截掉提示部分，只看生成
            attn = inputs["attention_mask"]
            prompt_lens = attn.sum(dim=1).tolist()
            for j in range(outputs.size(0)):
                gen_ids = outputs[j, prompt_lens[j]:]
                text = tok.decode(gen_ids, skip_special_tokens=True)
                cleaned, choice = clean_and_extract(text)
                preds.append((choice or "").upper())

        correct = sum(int(p == g) for p, g in zip(preds, golds))
        acc = correct / len(golds) if golds else 0.0
        total_correct += correct; total_cnt += len(golds)
        print(f"{sub:28s} {len(golds):4d} {acc:6.2%}")

    if total_cnt:
        print(f"\nOverall: {total_correct}/{total_cnt} = {total_correct/total_cnt:.2%}")
    else:
        print("\nOverall: NA")
        
if __name__ == "__main__":
    main()
