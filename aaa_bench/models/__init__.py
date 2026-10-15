"""The seven models evaluated in the paper, pinned to the exact checkpoints used."""

import importlib

# name -> (module, class, Hugging Face repository, commit)
MODELS = {
    "llava-1.5-7b": ("llava", "LLaVA", "llava-hf/llava-1.5-7b-hf", "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"),
    "llava-next-7b": ("llava", "LLaVA", "llava-hf/llava-v1.6-mistral-7b-hf", "2424fdd47412fccc66d91719126b420e9fbd7065"),
    "qwen2-vl-7b": ("qwen_vl", "QwenVL", "Qwen/Qwen2-VL-7B-Instruct", "eed13092ef92e448dd6875b2a00151bd3f7db0ac"),
    "qwen2.5-vl-7b": ("qwen_vl", "QwenVL", "Qwen/Qwen2.5-VL-7B-Instruct", "cc594898137f460bfe9f0759e9844b3ce807cfb5"),
    "internvl3-8b": ("internvl", "InternVL", "OpenGVLab/InternVL3-8B", "853e3a797a661694b1b8ece0cb72dc2b23e3dac9"),
    "phi-3.5-vision": ("phi3v", "Phi3Vision", "microsoft/Phi-3.5-vision-instruct", "12b77fb40b63a2c73c68243d3f767aab688a1b2a"),
    "idefics3-8b": ("idefics3", "Idefics3", "HuggingFaceM4/Idefics3-8B-Llama3", "fddb4ff79181e55a994674777e06cd5456ce3dc3"),
}


def get_model(name, revision=None):
    """A wrapper for one of the evaluated models. `revision=None` uses the commit from the paper."""
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; available: {', '.join(MODELS)}")
    module, cls_name, hf_id, commit = MODELS[name]
    cls = getattr(importlib.import_module(f"{__name__}.{module}"), cls_name)
    model = cls(name)
    model.hf_id, model.revision = hf_id, revision or commit
    return model
