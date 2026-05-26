import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from verl_bridge.chat_template_utils import apply_chat_template_compat


class BaseLLM:
    def __init__(self, model_name: str, device="cuda", trainable=False):
        self.device = device
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=os.getenv("TRANSFORMERS_CACHE"))
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True, cache_dir=os.getenv("TRANSFORMERS_CACHE")).to(device)
        if not trainable:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def generate(self, prompt: str, max_new_tokens=128, temperature=0.8, top_p=0.9):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self._eos_ids(),
            )

        gen_ids = outputs[0][input_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text

    def _eos_ids(self):
        ids = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []
        for tok in ["<|im_end|>", "<|endoftext|>"]:
            if tok in self.tokenizer.get_vocab():
                ids.append(self.tokenizer.convert_tokens_to_ids(tok))
        # remove duplicates
        return list({i for i in ids if i is not None})

    def generate_chat(self, system_prompt: str, user_prompt: str, **kw):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        chat_text = apply_chat_template_compat(
            self.tokenizer, messages, tokenize=False, add_generation_prompt=True
        )
        return self.generate(chat_text, **kw)
