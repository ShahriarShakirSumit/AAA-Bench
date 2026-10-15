"""Qwen2-VL-7B-Instruct and Qwen2.5-VL-7B-Instruct."""

import torch
from PIL import Image

from .base import VLM


class QwenVL(VLM):
    def load_model(self):
        from transformers import AutoProcessor

        if "2.5" in self.hf_id:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
        else:
            from transformers import Qwen2VLForConditionalGeneration as ModelClass
        self._model = ModelClass.from_pretrained(self.hf_id, revision=self.revision, torch_dtype=self.torch_dtype,
                                                 device_map="auto", low_cpu_mem_usage=True)
        self._processor = AutoProcessor.from_pretrained(self.hf_id, revision=self.revision)

    def generate(self, image_path, prompt, max_new_tokens=512, temperature=0.0):
        if self._model is None:
            self.load_model()
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[image], return_tensors="pt", padding=True).to(self._model.device)
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                                              temperature=temperature if temperature > 0 else None)
        return self._decode_new_tokens(output_ids, inputs)
