"""Compatibility patches for running the models' remote code on Transformers 5.

InternVL3 and Phi-3.5-Vision ship their own modelling code, written against Transformers 4. Under
Transformers 5 that code expects attributes and helpers that no longer exist. `apply()` restores
them before any model is loaded. It does nothing on Transformers 4. These are the patches in force
for every run reported in the paper.
"""

import inspect


def apply():
    import transformers

    if int(transformers.__version__.split(".")[0]) < 5:
        return

    import transformers.utils.import_utils as import_utils
    from transformers import DynamicCache, GenerationConfig, GenerationMixin, PreTrainedModel

    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: False

    # Older cache API used by remote code.
    init = DynamicCache.__init__

    def cache_init(self, *a, **k):
        init(self, *a, **k)
        if not hasattr(self, "seen_tokens"):
            self.seen_tokens = 0

    DynamicCache.__init__ = cache_init
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: getattr(self, "max_cache_len", None)
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = (
            lambda self, n, layer_idx=0: self.get_seq_length(layer_idx if isinstance(layer_idx, int) else 0))

    # Remote-code language models expect .generate() and .generation_config on every PreTrainedModel.
    if GenerationMixin not in PreTrainedModel.__bases__:
        PreTrainedModel.__bases__ = (*PreTrainedModel.__bases__, GenerationMixin)
    from_pretrained = PreTrainedModel.from_pretrained

    @classmethod
    def from_pretrained_with_config(cls, *a, **k):
        model = from_pretrained.__func__(cls, *a, **k)
        if not hasattr(model, "generation_config"):
            model.generation_config = GenerationConfig()
        return model

    PreTrainedModel.from_pretrained = from_pretrained_with_config

    # Device-map inference reads all_tied_weights_keys, which remote-code models never set.
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}
    import transformers.integrations.accelerate as hf_accelerate

    if hasattr(hf_accelerate, "_init_infer_auto_device_map"):
        infer = hf_accelerate._init_infer_auto_device_map

        def infer_with_tied_keys(model, *a, **k):
            if not hasattr(model, "all_tied_weights_keys"):
                model.__dict__["all_tied_weights_keys"] = {}
            return infer(model, *a, **k)

        hf_accelerate._init_infer_auto_device_map = infer_with_tied_keys

    # tie_weights() in older remote code does not accept the keyword arguments Transformers 5 passes.
    import transformers.modeling_utils as modeling_utils

    if hasattr(modeling_utils.PreTrainedModel, "_finalize_model_loading"):
        finalize = modeling_utils.PreTrainedModel._finalize_model_loading

        @staticmethod
        def finalize_with_old_tie_weights(model, *a, **k):
            sig = inspect.signature(model.__class__.tie_weights)
            takes_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
            if "missing_keys" not in sig.parameters and not takes_kwargs:
                tie = model.tie_weights

                def tie_weights(*aa, **kk):
                    kk.pop("missing_keys", None)
                    kk.pop("recompute_mapping", None)
                    return tie(*aa, **kk)

                model.tie_weights = tie_weights
            return finalize(model, *a, **k)

        modeling_utils.PreTrainedModel._finalize_model_loading = finalize_with_old_tie_weights
