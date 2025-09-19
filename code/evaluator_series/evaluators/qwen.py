# evaluators/qwen.py
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

    @ torch.no_grad()
    def _first_step_logits_choice(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen = self.model.generate(**inputs, do_sample=False, max_new_tokens=1,
                                  return_dict_in_generate=True, output_scores=True)
        scores = gen.scores[0][0]                 # (vocab_size,)
        letter_ids = [self.tokenizer.encode(x, add_special_tokens=False)[0] for x in ["A","B","C","D"]]
        pred_index = int(np.argmax(scores[letter_ids].float().cpu().numpy()))
        return self.choices[pred_index]

