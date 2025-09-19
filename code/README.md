# README

Qwen3-0.6B × C-Eval 离线评测（Parquet/CSV 兼容）

本仓库实现 Qwen3-0.6B（Base） 在 C-Eval 数据集上的离线评测。
默认策略是 logits 直选（不采样，A/B/C/D 四选一首 token 最高），可复现、速度快。
说明：这套流程是“评测/测试”，不是训练/微调。

目录结构（示例）
offline_bundle/
  ├─ model/Qwen3-0.6B/                 # 本地模型权重（也可用 HF 远程ID）
  └─ ceval/ceval-exam/                 # C-Eval 数据根（Parquet 或 CSV）
      ├─ high_school_biology/val-00000-of-00001.parquet
      ├─ ...
qwen_training/
  ├─ subject_mapping.json              # 学科映射（可选，用于分组/筛选）
  └─ code/evaluator_series/
      ├─ eval.py                       # 单学科评测脚本（CSV/Parquet 自适应）
      └─ subjects.txt                  # 自定义学科清单（可选）

环境准备
# 推荐：Miniforge + conda
module load Miniforge3
conda create -n qwen312 python=3.12 -y
conda activate qwen312

# PyTorch (CUDA 12.4)
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.6.0" "torchvision==0.21.0" triton

# 其余依赖
python -m pip install "transformers==4.51.0" "pandas==2.3.2" pyarrow tqdm \
  sentencepiece tokenizers accelerate huggingface_hub


你的集群已验证：torch 2.6.0+cu124 / transformers 4.51.0 / pandas 2.3.2 / pyarrow 可用。
Parquet 数据必须安装 pyarrow。

数据路径
# 指向 ceval-exam 那一层（支持学科式和聚合式；CSV/Parquet 都行）
export CEVAL_DATA_DIR=~/offline_bundle/ceval/ceval-exam

快速开始（单学科）
cd ~/offline_bundle/qwen_training
module load Miniforge3 && conda activate qwen312
export PYTHONPATH=$PWD:$PYTHONPATH

cd code/evaluator_series
python eval.py --model_name ~/offline_bundle/model/Qwen3-0.6B \
               --subject high_school_biology
# 示例输出：
# [INFO] Using val data for 'high_school_biology': [.../val-00000-of-00001.parquet]
# Acc: 36.84


自定义清单 & 仅在终端打印成绩

1）写清单（示例：高中科目）

cat > subjects.txt <<'TXT'
# 我想跑的科目
high_school_biology
high_school_chemistry
high_school_physics
high_school_mathematics
high_school_politics
high_school_geography
high_school_history
high_school_chinese
TXT


2）循环跑并只在终端打印（不落文件）

module load Miniforge3 && conda activate qwen312
export CEVAL_DATA_DIR=~/offline_bundle/ceval/ceval-exam
export PYTHONPATH=$(git rev-parse --show-toplevel):$PYTHONPATH

printf "%-26s %s\n" "subject" "acc"
printf "%-26s %s\n" "-------" "----"
while read -r s; do
  [[ -z "$s" || "$s" =~ ^# ]] && continue
  acc=$(python eval.py --model_name ~/offline_bundle/model/Qwen3-0.6B \
                       --subject "$s" 2>&1 | awk '/^Acc:/{a=$2} END{print a}')
  printf "%-26s %s\n" "$s" "${acc:-NA}"
done < subjects.txt


一键把高中清单替换成初中版：

sed -E 's/^(high_school_)/middle_school_/' subjects.txt > subjects_middle_school.txt

可选策略与小优化

更稳的打分：将“首 token 直选”升级为“完整选项对数似然”评分（可加一个开关实现）。

生成式评测：换 Qwen/Qwen3-0.6B-Instruct 并开启 --few_shot/--cot，用正则抽答案。

去除采样参数警告：在 evaluators/qwen.py::__init__ 加入：

self.model.eval()
cfg = self.model.generation_config
cfg.do_sample = False
cfg.temperature = None
cfg.top_p = None
cfg.top_k = None


这样 generate(..., do_sample=False) 就不会提示 temperature/top_p/top_k 的警告。

常见问题（FAQ）

找不到数据
确认 CEVAL_DATA_DIR 指向 .../ceval-exam，且存在
subject/val.csv 或 subject/val-*.parquet（或聚合式 val/<subject>_val.csv）。

Parquet 报错
安装 pyarrow：pip install -U pyarrow。

成绩波动大
单科学样本量常见只有 ~18–20，波动正常；看整组（如 High School）均值更稳定。

提交到 GitHub
结果建议放到 results/（非 outputs/），再 git add results/*.csv && git commit && git push。
推送失败常见是未配置 SSH key 或 PAT。

致谢（来源标注）

模型：Qwen3-0.6B（Base/Instruct），来自 Qwen 团队（Hugging Face）。

数据集：C-Eval（HKUST-NLP）。

代码：在 CEval 评测思路基础上，改写为 CSV/Parquet 自适应 + logits 直选，并新增批量评测脚本。

English (Quick)

What: Offline evaluation of Qwen3-0.6B (Base) on C-Eval, with a logits-only MCQ scorer. Parquet/CSV supported; batch evaluation loads model once and writes CSV summaries.

Single subject

cd ~/offline_bundle/qwen_training
module load Miniforge3 && conda activate qwen312
export PYTHONPATH=$PWD:$PYTHONPATH
export CEVAL_DATA_DIR=~/offline_bundle/ceval/ceval-exam

cd code/evaluator_series
python eval.py --model_name ~/offline_bundle/model/Qwen3-0.6B --subject high_school_biology



Custom list, print-only

printf "%-26s %s\n" "subject" "acc"
while read -r s; do
  [[ -z "$s" || "$s" =~ ^# ]] && continue
  acc=$(python eval.py --model_name ~/offline_bundle/model/Qwen3-0.6B \
                       --subject "$s" 2>&1 | awk '/^Acc:/{a=$2} END{print a}')
  printf "%-26s %s\n" "$s" "${acc:-NA}"
done < subjects.txt


Notes: default split is val; install pyarrow for Parquet; silence sampling warnings via generation_config as shown above.