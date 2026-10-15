"""Phi-3.5-Vision-Instruct (remote code, eager attention)."""

import torch
from PIL import Image

from .base import VLM


class Phi3Vision(VLM):
    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._model = AutoModelForCausalLM.from_pretrained(
            self.hf_id, revision=self.revision, torch_dtype=self.torch_dtype, device_map="auto",
            trust_remote_code=True, _attn_implementation="eager")
        self._processor = AutoProcessor.from_pretrained(self.hf_id, revision=self.revision, trust_remote_code=True)

    def generate(self, image_path, prompt, max_new_tokens=512, temperature=0.0):
        if self._model is None:
            self.load_model()
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": f"<|image_1|>\n{prompt}"}]
        text = self._processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text, [image], return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                                              temperature=temperature if temperature > 0 else None,
                                              eos_token_id=self._processor.tokenizer.eos_token_id, use_cache=False)
        return self._decode_new_tokens(output_ids, inputs)
