"""InternVL3-8B (remote code), with the model card's dynamic tiling at 448 px."""

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from .base import VLM

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if min_num <= i * j <= max_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    tiles = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        tiles.append(resized_img.crop(box))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


class InternVL(VLM):
    def load_model(self):
        from transformers import AutoModel, AutoTokenizer, GenerationConfig, GenerationMixin

        self._model = AutoModel.from_pretrained(
            self.hf_id, revision=self.revision, torch_dtype=self.torch_dtype, device_map="auto",
            low_cpu_mem_usage=True, trust_remote_code=True, attn_implementation="eager")
        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_id, revision=self.revision,
                                                        trust_remote_code=True, use_fast=False)
        self._model.eval()
        # Under Transformers 5 the remote InternLM/Qwen language model lacks .generate() and a
        # generation config; restore both so the model's own chat() works.
        lm = self._model.language_model
        if not hasattr(lm, "generate"):
            lm.__class__.__bases__ = (*lm.__class__.__bases__, GenerationMixin)
        if not hasattr(lm, "generation_config"):
            lm.generation_config = GenerationConfig()
        if not hasattr(self._model, "generation_config"):
            self._model.generation_config = GenerationConfig()

    def generate(self, image_path, prompt, max_new_tokens=512, temperature=0.0):
        if self._model is None:
            self.load_model()
        image = Image.open(image_path).convert("RGB")
        transform = build_transform(448)
        tiles = dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=12)
        pixel_values = torch.stack([transform(t) for t in tiles]).to(self.torch_dtype).cuda()
        config = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            config["temperature"] = temperature
        return self._model.chat(self._tokenizer, pixel_values, prompt, config).strip()
