from __future__ import annotations
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
from pathlib import Path
import torch


class DINOv3Extractor:
    def __init__(self, model_dir: str | Path, image_size: int = 448, amp: str = "bf16"):
        import keras
        import keras_hub
        self.keras = keras
        self.model_dir = str(model_dir)
        self.image_size = image_size
        self.amp = amp
        if torch.cuda.is_available() and amp == "bf16" and torch.cuda.is_bf16_supported():
            keras.mixed_precision.set_global_policy("mixed_bfloat16")
        elif torch.cuda.is_available() and amp == "fp16":
            keras.mixed_precision.set_global_policy("mixed_float16")
        else:
            keras.mixed_precision.set_global_policy("float32")
        device_name = "gpu:0" if torch.cuda.is_available() else "cpu:0"
        with keras.device(device_name):
            backbone = keras_hub.models.DINOV3Backbone.from_preset(
                self.model_dir, image_shape=(image_size, image_size, 3)
            )
        self.backbone = backbone
        self.num_layers = int(backbone.num_layers)
        self.hidden_dim = int(backbone.hidden_dim)
        self.num_prefix = 1 + int(backbone.num_register_tokens)
        outputs = {f"stage{i}": backbone.pyramid_outputs[f"stage{i}"] for i in range(1, self.num_layers + 1)}
        self.model = keras.Model(inputs=backbone.inputs, outputs=outputs)
        self.model.trainable = False

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor) -> dict[int, torch.Tensor]:
        # Keras torch backend accepts torch BHWC. Put tensor on CUDA before call.
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
        out = self.model({"pixel_values": images}, training=False)
        result = {}
        for i in range(1, self.num_layers + 1):
            t = out[f"stage{i}"]
            if not isinstance(t, torch.Tensor):
                t = torch.as_tensor(t)
            t = t[:, self.num_prefix:, :].float()
            result[i] = t
        return result

    def pooled(self, feats: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        return {k: torch.nn.functional.normalize(v.mean(dim=1), dim=-1) for k,v in feats.items()}
