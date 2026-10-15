"""LLaVA-1.5-7B and LLaVA-NeXT-7B (Mistral)."""

import torch
from PIL import Image

from .base import VLM


class LLaVA(VLM):
    def load_model(self):
        from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaNextForConditionalGeneration

        cls = LlavaNextForConditionalGeneration if "v1.6" in self.hf_id else LlavaForConditionalGeneration
        self._model = cls.from_pretrained(self.hf_id, revision=self.revision, torch_dtype=self.torch_dtype,
                                          device_map="auto", low_cpu_mem_usage=True)
        self._processor = AutoProcessor.from_pretrained(self.hf_id, revision=self.revision)

    def generate(self, image_path, prompt, max_new_tokens=512, temperature=0.0):
        if self._model is None:
            self.load_model()
        image = Image.open(image_path).convert("RGB")
        conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self._processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self._processor(images=image, text=text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                                              temperature=temperature if temperature > 0 else None)
        return self._decode_new_tokens(output_ids, inputs)
