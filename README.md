Qwen3 CEval Evaluator (HF)

Origin / 代码来源标注
    本仓库的评测思路与部分结构参考并改写自 HKUST-NLP/C-Eval（原仓库包含 LLaMA 等评测器与数据说明）。
    我们将其中“按 A/B/C/D 选项进行比较”的评测方式适配到了 HuggingFace Transformers，并新增对 Qwen3 等 HF 格式模型的直接评测支持。
    数据集版权归原作者所有；本仓库不包含任何模型权重或数据文件。

Overview / 项目简介

  评测 Qwen3-0.6B（或其他 HF CausalLM）在 C-Eval 数据集上的选择题准确率。

  支持 logits 直选（对 A/B/C/D 的下一步 token 概率进行比较，零样本更稳）与 生成式解析 两种模式。


Repo Structure / 仓库结构（示例）
           
├─ code/
│  └─ evaluator_series/
│     ├─ evaluators/
│     │  ├─ qwen.py            
│     │  └─ evaluator.py       
│     └─ eval.py               
├─ ceval/   
├─ outputs/                    
├─ .gitignore
└─ README.md

Setup / 环境准备
    
    module load Miniforge3
    conda create -n qwen312 python=3.12 -y
    conda activate qwen312
    pip install -U pip
    
    # PyTorch（CUDA 12.4）
    pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
      "torch==2.6.0" "torchvision==0.21.0" triton
    
    # 其余依赖
    pip install "transformers==4.51.0" tokenizers accelerate pandas tqdm sentencepiece huggingface_hub

Data & Model / 数据与模型

下载 C-Eval 数据（学科式目录）：
    
    # 会得到 ceval-exam/<subject>/{dev,val,test}.csv
    huggingface-cli download --repo-type dataset ceval/ceval-exam \
      --local-dir ./ceval/ceval-exam --local-dir-use-symlinks False


subject 映射表：
    
    curl -L -o ./ceval/subject_mapping.json \
      "https://raw.githubusercontent.com/hkust-nlp/ceval/main/subject_mapping.json"
    

模型（例：Qwen3-0.6B）：
    
    huggingface-cli download Qwen/Qwen3-0.6B \
      --local-dir ./model/Qwen3-0.6B --local-dir-use-symlinks False


快速开始（单学科）:
	
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
	middle_school_biology
	middle_school_chemistry
	middle_school_physics
	middle_school_mathematics
	middle_school_politics
	middle_school_geography
	middle_school_history
	middle_school_chinese
	TXT


2）循环跑并只在终端打印（不落文件）
	
	module load Miniforge3 && conda activate qwen312
	export CEVAL_DATA_DIR=~/offline_bundle/ceval/ceval-exam
	export PYTHONPATH=$(git rev-parse --show-toplevel):$PYTHONPATH
	
	#zero-shot
	printf "%-26s %s\n" "subject" "acc"
	printf "%-26s %s\n" "-------" "----"
	while read -r s; do
	  [[ -z "$s" || "$s" =~ ^# ]] && continue
	  acc=$(python eval.py --model_name ~/offline_bundle/model/Qwen3-0.6B \
	                       --subject "$s" 2>&1 | awk '/^Acc:/{a=$2} END{print a}')
	  printf "%-26s %s\n" "$s" "${acc:-NA}"
	done < subjects.txt
	
	#few-shot
	
	printf "%-26s %s\n" "subject" "acc"
	printf "%-26s %s\n" "-------" "----"
	while read -r s; do
	  [[ -z "$s" || "$s" =~ ^# ]] && continue
	  acc=$(python eval.py --model_name ~/offline_bundle/model/Qwen3-0.6B \
	                       --subject "$s" --few_shot -k 5 2>&1 \
	        | awk '/^Acc:/{a=$2} END{print a}')
	  printf "%-26s %s\n" "$s" "${acc:-NA}"
	done < subjects.txt
