"""Idefics3-8B-Llama3, run in bfloat16."""

import torch
from PIL import Image

from .base import VLM


class Idefics3(VLM):
    def __init__(self, name, dtype="bfloat16"):
        super().__init__(name, dtype)

    def load_model(self):
        from transformers import AutoProcessor, Idefics3ForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(self.hf_id, revision=self.revision)
        self._model = Idefics3ForConditionalGeneration.from_pretrained(
            self.hf_id, revision=self.revision, torch_dtype=torch.bfloat16, device_map="auto")

    def generate(self, image_path, prompt, max_new_tokens=512, temperature=0.0):
        if self._model is None:
            self.load_model()
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(images=image, text=text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                                              temperature=temperature if temperature > 0 else None)
        return self._decode_new_tokens(output_ids, inputs)
