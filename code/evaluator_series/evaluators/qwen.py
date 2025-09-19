# evaluators/qwen.py
# 在仓库根或 evaluator_series 目录均可运行：
import os, re, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .evaluator import Evaluator   # 同目录下已有

class Qwen_Evaluator(Evaluator):
    def __init__(self, choices, k, model_name: str, device: torch.device | None = None,
                 dtype: str = "fp16"):
        super().__init__(choices, model_name, k)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype == "fp16": _dtype = torch.float16
        elif dtype == "bf16": _dtype = torch.bfloat16
        else: _dtype = torch.float32

        # 允许本地目录或HF模型ID（Base/Instruct均可）
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=_dtype, device_map="auto"
        )
        # pad/eos
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.eval()

        # 关闭全局 sampling 警告；具体生成时再显式传参
        cfg = self.model.generation_config
        cfg.do_sample = False; cfg.temperature = None; cfg.top_p = None; cfg.top_k = None

        # 从生成文本抽取答案的正则（中文/中英混排常见写法）
        self.answer_patterns = [
            r"所以答案是\s*([ABCD])", r"答案为\s*([ABCD])", r"答案是\s*([ABCD])",
            r"选择\s*([ABCD])", r"答案：\s*([ABCD])", r"选项\s*([ABCD])\s*正确",
            r"最终答案\s*[:：]?\s*([ABCD])", r"\b([ABCD])\b\s*是正确的"
        ]

    # 单题格式化
    def format_example(self, line, include_answer: bool = True, cot: bool = False) -> str:
        example = line['question']
        for ch in self.choices:
            example += f"\n{ch}. {line[f'{ch}']}"
        if include_answer:
            if cot:
                expl = (line.get("explanation") or "").strip()
                if not expl: expl = "略。"
                example += f"\n答案：让我们一步一步思考，\n{expl}\n所以答案是{line['answer']}。\n\n"
            else:
                example += f"\n答案：{line['answer']}\n\n"
        else:
            if cot:
                example += "\n答案：让我们一步一步思考，\n1."
            else:
                example += "\n答案："
        return example

    # few-shot 前缀
    def generate_few_shot_prompt(self, subject: str, dev_df, cot: bool = False) -> str:
        prompt = f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n"
        k = dev_df.shape[0] if self.k == -1 else min(self.k, dev_df.shape[0])
        for i in range(k):
            prompt += self.format_example(dev_df.iloc[i, :], include_answer=True, cot=cot)
        return prompt

    @torch.no_grad()
    def _first_step_logits_choice(self, prompt: str) -> str:
        # 对“答案：”后的第一个生成token做 argmax，取 A/B/C/D 中分数最高者
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen = self.model.generate(
            **inputs, do_sample=False, max_new_tokens=1,
            return_dict_in_generate=True, output_scores=True
        )
        scores = gen.scores[0][0]  # (vocab_size,)
        letter_ids = [self.tokenizer.encode(x, add_special_tokens=False)[0] for x in ["A","B","C","D"]]
        pred_index = int(torch.argmax(scores[letter_ids]).item())
        return self.choices[pred_index]

    def _extract_answer_from_text(self, text: str) -> str | None:
        for pat in self.answer_patterns:
            m = re.search(pat, text, re.M)
            if m: return m.group(1)
        letters = re.findall(r"[ABCD]", text)
        if len(letters) == 1: return letters[0]
        return None

    @torch.no_grad()
    def _generate_text(self, prompt: str, max_new_tokens=128, temperature=0.2, top_p=0.9, do_sample=True) -> str:
        # few-shot/CoT 推荐适度采样；如需完全确定性可把 do_sample=False
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            do_sample=do_sample, temperature=temperature, top_p=top_p,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def eval_subject(self, subject_name, test_df, dev_df=None, few_shot=False, cot=False, save_result_dir=None):
        results, scores = [], []
        if few_shot:
            assert dev_df is not None and len(dev_df) > 0, "few_shot 需要提供 dev_df"
            prefix = self.generate_few_shot_prompt(subject_name, dev_df, cot=cot)

        answers = list(test_df['answer'])
        for i, row in enumerate(test_df.itertuples(index=False)):
            line = row._asdict() if hasattr(row, "_asdict") else test_df.iloc[i, :]

            if few_shot:
                q = self.format_example(line, include_answer=False, cot=cot)
                full_prompt = prefix + q
                text = self._generate_text(full_prompt, max_new_tokens=256,
                                           temperature=0.2 if not cot else 0.7,
                                           top_p=0.9, do_sample=True)
                completion = text[len(full_prompt):]
                pred = self._extract_answer_from_text(completion)
                if pred is None:
                    # 回退：用 logits 直选提高鲁棒性
                    pred = self._first_step_logits_choice(full_prompt)
            else:
                # zero-shot logits 直选
                q = self.format_example(line, include_answer=False, cot=False)
                full_prompt = q
                pred = self._first_step_logits_choice(full_prompt)

            correct = 1 if pred == answers[i] else 0
            results.append(pred); scores.append(correct)

        acc_pct = 100 * sum(scores) / len(scores)
        if save_result_dir:
            os.makedirs(save_result_dir, exist_ok=True)
            out_df = test_df.copy()
            out_df['model_output'] = results
            out_df['correctness'] = scores
            out_df.to_csv(os.path.join(save_result_dir, f'{subject_name}_test.csv'), index=False, encoding="utf-8")
        return acc_pct