Qwen3 CEval Evaluator (HF)

    基于 HuggingFace 的 Qwen3 C-Eval 评测脚手架

Origin / 代码来源标注
    本仓库的评测思路与部分结构参考并改写自 HKUST-NLP/C-Eval（原仓库包含 LLaMA 等评测器与数据说明）。
    我们将其中“按 A/B/C/D 选项进行比较”的评测方式适配到了 HuggingFace Transformers，并新增对 Qwen3 等 HF 格式模型的直接评测支持。
    数据集版权归原作者所有；本仓库不包含任何模型权重或数据文件。

Overview / 项目简介

    评测 Qwen3-0.6B（或其他 HF CausalLM）在 C-Eval 数据集上的选择题准确率。
    
    支持 logits 直选（对 A/B/C/D 的下一步 token 概率进行比较，零样本更稳）与 生成式解析 两种模式。

兼容两种数据目录布局：
    
    val/<subject>_val.csv（聚合式）
    
    <subject>/val.csv（学科式，Hugging Face 下载的默认结构）
    
    可仅评 C-level（大学水平）或指定学科子集；可离线运行。
    
    提供 Slurm 批处理示例；支持在 HPC 上一键复现实验。

Features / 特性

HF 模型即插即用（本地目录或 org/name 在线加载）

    --c_level_only 一键筛选大学层级科目（若筛不到会自动回退到“全部有 CSV 的科目”，避免 n=0）

    --mode logits|generate 切换评测策略

    --offline 完全离线（需提前准备模型与数据）

    输出总体与分学科准确率（JSON）

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

Acknowledgements & License / 致谢与许可
    
    C-Eval：题目设计与数据版权归原作者 HKUST-NLP/C-Eval
     所有，使用请遵循其上游许可与使用条款。
    
    Qwen：模型版权归其作者与组织所有；请遵循其对应仓库与模型卡的许可。
    
    Transformers & PyTorch：本项目依赖 HuggingFace Transformers
     与 PyTorch
    ，请遵循其许可。
    
    本仓库仅提供评测代码，不包含任何受版权保护的数据或权重；如需发布结果或复现实验，请在论文/报告中对 C-Eval 与相应模型进行明确引用与致谢。
