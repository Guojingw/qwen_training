# evaluators/qwen.py
import os
import re, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluators.evaluator import Evaluator

class Qwen_Evaluator(Evaluator):
    def __init__(self, choices, k, model_name: str, device: torch.device | None = None,
                 dtype: str = "fp16"):
        super().__init__(choices, model_name, k)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _dtype = torch.float16 if dtype=="fp16" else (torch.bfloat16 if dtype=="bf16" else torch.float32)

        # 允许本地目录或HF模型名
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=_dtype, device_map="auto"
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.answer_patterns = [
            r"所以答案是\s*([ABCD])", r"答案为\s*([ABCD])", r"答案是\s*([ABCD])",
            r"选择\s*([ABCD])", r"答案：\s*([ABCD])", r"选项\s*([ABCD])\s*正确", r"([ABCD])\s*是正确的",
        ]

    def format_example(self, line, include_answer: bool = True, cot: bool = False) -> str:
        example = line['question']
        for choice in self.choices:
            example += f"\n{choice}. {line[f'{choice}']}"
        if include_answer:
            if cot:
                example += "\n答案：让我们一步一步思考，\n" + line["explanation"] + f"\n所以答案是{line['answer']}。\n\n"
            else:
                example += "\n答案：" + line["answer"] + "\n\n"
        else:
            if cot:
                example += "\n答案：让我们一步一步思考，\n1."
            else:
                example += "\n答案："
        return example

    def generate_few_shot_prompt(self, subject: str, dev_df, cot: bool = False) -> str:
        prompt = f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n"
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            prompt += self.format_example(dev_df.iloc[i, :], include_answer=True, cot=cot)
        return prompt
    
        def _generate_text(self, prompt: str, max_new_tokens: int = 256,
                       temperature: float = 0.2, top_p: float = 0.9) -> str:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def _extract_answer_from_text(self, text: str) -> str | None:
        for pat in self.answer_patterns:
            m = re.search(pat, text, re.M)
            if m:
                return m.group(1)
        hits = re.findall(r"[ABCD]", text)
        return hits[0] if len(hits) == 1 else None

    def eval_subject(self, subject_name, test_df, dev_df=None, few_shot=False,
                     save_result_dir=None, cot: bool = False):
        """返回正确率(百分比)并可保存 per-question 结果到 CSV。"""
        results, scores = [], []

        # few-shot 头部提示
        few_shot_prompt = ""
        if few_shot:
            assert dev_df is not None, "few_shot=True 需要传入 dev_df"
            few_shot_prompt = self.generate_few_shot_prompt(subject_name, dev_df, cot=cot)

        answers = list(test_df["answer"])
        for idx, row in test_df.iterrows():
            question = self.format_example(row, include_answer=False, cot=cot)
            full_prompt = few_shot_prompt + question

            if few_shot or cot:
                # 生成式：完整生成，再从文本里抽答案
                gen_text = self._generate_text(full_prompt,
                                               max_new_tokens=256,
                                               temperature=0.7 if cot else 0.2,
                                               top_p=0.9)
                completion = gen_text[len(full_prompt):]
                pred = self._extract_answer_from_text(completion) or "-"
            else:
                # 零样本：用 A/B/C/D 的“首 token logits”择优
                pred = self._first_step_logits_choice(full_prompt)

            gold = str(answers[idx]).strip().upper()[:1]
            correct = 1 if pred == gold else 0
            results.append(pred); scores.append(correct)

        acc = 100.0 * sum(scores) / len(scores) if len(scores) else 0.0

        if save_result_dir:
            os.makedirs(save_result_dir, exist_ok=True)
            out_df = test_df.copy()
            out_df["model_output"] = results
            out_df["correctness"] = scores
            out_df.to_csv(os.path.join(save_result_dir, f"{subject_name}_test.csv"),
                          index=False, encoding="utf-8")
        return acc


    @ torch.no_grad()
    def _first_step_logits_choice(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen = self.model.generate(**inputs, do_sample=False, max_new_tokens=1,
                                  return_dict_in_generate=True, output_scores=True)
        scores = gen.scores[0][0]                 # (vocab_size,)
        letter_ids = [self.tokenizer.encode(x, add_special_tokens=False)[0] for x in ["A","B","C","D"]]
        pred_index = int(np.argmax(scores[letter_ids].float().cpu().numpy()))
        return self.choices[pred_index]

