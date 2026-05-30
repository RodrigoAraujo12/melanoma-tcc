import gc
import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor
from huggingface_hub import login


MODEL_ID = "google/medgemma-4b-it"


def load_medgemma_vision(hf_token: str):
    login(token=hf_token)
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=hf_token)
    full_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        token=hf_token,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    vision_encoder = full_model.vision_tower
    vision_encoder.requires_grad_(False)
    try:
        del full_model.language_model
    except AttributeError:
        pass
    try:
        del full_model.multi_modal_projector
    except AttributeError:
        pass
    del full_model
    gc.collect()
    torch.cuda.empty_cache()
    return vision_encoder, processor


class DermClassifier(nn.Module):
    def __init__(self, vision_encoder: nn.Module, vision_hidden_size: int,
                 metadata_dim: int = 13, num_classes: int = 5,
                 freeze_vision: bool = True, hidden_dim: int = 256):
        super().__init__()
        self.vision_encoder = vision_encoder
        if freeze_vision:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
            self.vision_encoder.eval()

        self.vision_proj = nn.Sequential(
            nn.LayerNorm(vision_hidden_size),
            nn.Linear(vision_hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

        self._freeze_vision = freeze_vision

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        ctx = torch.no_grad() if self._freeze_vision else torch.enable_grad()
        with ctx:
            outputs = self.vision_encoder(pixel_values=pixel_values)
            if hasattr(outputs, "last_hidden_state"):
                features = outputs.last_hidden_state
            else:
                features = outputs[0]
            pooled = features.mean(dim=1)
        return pooled.float()

    def forward(self, pixel_values: torch.Tensor, metadata: torch.Tensor,
                labels: torch.Tensor = None):
        v_feat = self.encode_image(pixel_values)
        v_feat = self.vision_proj(v_feat)
        m_feat = self.metadata_encoder(metadata.float())
        combined = torch.cat([v_feat, m_feat], dim=-1)
        logits = self.classifier(combined)
        if labels is not None:
            return {"logits": logits}
        return logits


def build_dermclassifier(hf_token: str, num_classes: int = 5,
                         metadata_dim: int = 13, freeze_vision: bool = True):
    vision_encoder, processor = load_medgemma_vision(hf_token)
    hidden = vision_encoder.config.hidden_size
    model = DermClassifier(
        vision_encoder=vision_encoder,
        vision_hidden_size=hidden,
        metadata_dim=metadata_dim,
        num_classes=num_classes,
        freeze_vision=freeze_vision,
    )
    return model, processor
