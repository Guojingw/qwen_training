import os
import re
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluators.evaluator import Evaluator


class Qwen_Evaluator(Evaluator):
    def __init__(self, choices, k, model_name: str, device: torch.device | None = None):
        super(Qwen_Evaluator, self).__init__(choices, model_name, k)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Default to an instruct checkpoint name if a plain alias like "qwen3-0.6b" is given
        hf_model_name = model_name
        alias = model_name.lower().replace(" ", "")
        if alias in {"qwen3-0.6b", "qwen3-0.6b-instruct", "qwen3_0.6b", "qwen3_0.6b_instruct"}:
            # Adjust to the public HF repo name if needed
            # If this repo name differs in your env, change it to the exact one you have.
            hf_model_name = "Qwen/Qwen3-0.6B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None
        ).to(self.device)

        # Regex patterns to extract options from generated text when not using logits-based selection
        self.answer_patterns = [
            r"所以答案是\s*([ABCD])",
            r"答案为\s*([ABCD])",
            r"答案是\s*([ABCD])",
            r"选择\s*([ABCD])",
            r"答案：\s*([ABCD])",
            r"选项\s*([ABCD])\s*正确",
            r"([ABCD])\s*是正确的",
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

    def _generate_text(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text

    def _first_step_logits_choice(self, prompt: str) -> str:
        # Use the distribution of the first generated token to choose among A/B/C/D
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            gen = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
            )
        scores = gen.scores[0][0]  # (vocab_size,)
        # Map letters to token ids (best single-token match)
        letter_ids = []
        for letter in ["A", "B", "C", "D"]:
            token_ids = self.tokenizer.encode(letter, add_special_tokens=False)
            # Fallback if multiple tokens: take the first one
            letter_ids.append(token_ids[0])
        letter_scores = scores[letter_ids].detach().float().cpu().numpy()
        pred_index = int(np.argmax(letter_scores))
        return self.choices[pred_index]

    def _extract_answer_from_text(self, text: str) -> str | None:
        for pattern in self.answer_patterns:
            m = re.search(pattern, text, re.M)
            if m:
                return m.group(1)
        # fallback: if exactly one of A/B/C/D appears
        m_all = re.findall(r"[ABCD]", text)
        if len(m_all) == 1:
            return m_all[0]
        return None

    def eval_subject(self, subject_name, test_df, dev_df=None, few_shot=False, cot=False, save_result_dir=None):
        results: List[str] = []
        scores: List[int] = []
        few_shot_prompt = self.generate_few_shot_prompt(subject_name, dev_df, cot=cot) if few_shot else ""

        answers = list(test_df['answer'])
        for row_index, row in tqdm(test_df.iterrows(), total=len(test_df)):
            question = self.format_example(row, include_answer=False, cot=cot)
            full_prompt = few_shot_prompt + question

            if few_shot or cot:
                # Generate full text then extract
                gen_text = self._generate_text(full_prompt, max_new_tokens=256, temperature=0.2 if not cot else 0.7, top_p=0.9)
                # Only keep the model completion part after the prompt, for reliable extraction
                completion = gen_text[len(full_prompt):]
                pred = self._extract_answer_from_text(completion) or "-"
            else:
                # Zero-shot, pick by logits among A/B/C/D
                pred = self._first_step_logits_choice(full_prompt)

            correct = 1 if pred == answers[row_index] else 0
            results.append(pred)
            scores.append(correct)

        correct_ratio = 100 * sum(scores) / len(scores)

        if save_result_dir:
            test_df['model_output'] = results
            test_df['correctness'] = scores
            test_df.to_csv(os.path.join(save_result_dir, f'{subject_name}_test.csv'), encoding="utf-8", index=False)

        return correct_ratio


