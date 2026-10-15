"""Common interface of the model wrappers."""

from abc import ABC, abstractmethod

import torch


class VLM(ABC):
    """A vision-language model that maps (image path, prompt) to text, greedily."""

    hf_id = None        # Hugging Face repository
    revision = None     # the exact commit evaluated in the paper

    def __init__(self, name, dtype="float16"):
        self.name = name
        self.dtype = dtype
        self._model = None
        self._processor = None

    @property
    def model_name(self):
        return self.name

    @property
    def torch_dtype(self):
        return torch.float16 if self.dtype == "float16" else torch.bfloat16

    @abstractmethod
    def load_model(self):
        ...

    @abstractmethod
    def generate(self, image_path, prompt, max_new_tokens=512, temperature=0.0):
        ...

    def _decode_new_tokens(self, output_ids, inputs):
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._processor.decode(generated, skip_special_tokens=True).strip()
