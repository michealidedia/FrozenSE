"""
Pathology foundation-model encoders with the FrozenSE-SAM decoder (v1.6).

    pfm_phikon : Phikon, ViT-B/16 trained with iBOT on TCGA tiles (owkin/phikon).
                 Public, no account approval needed.
    pfm_uni    : UNI, ViT-L/16 (MahmoodLab/uni). Gated: needs an approved
                 Hugging Face account and HF_TOKEN in the environment.

Plain ViTs have one feature resolution (1/16: 32 x 32 for a 512 patch), so
the encoder-to-decoder interface is adapted, UNETR-style:
    neck  : last tokens -> 1x1 conv to 256 ch -> 2x transposed conv -> 64 x 64,
            the grid the decoder receives from MobileSAM's neck;
    skips : tokens after blocks depth/3 and 2*depth/3, projected to 64 and 128
            channels and upsampled to 256 x 256 and 128 x 128, the sizes and
            channel counts of MobileSAM's two skip features.
The adapters are new weights and train with the decoder (10 x learning rate).

Inputs arrive normalised with ImageNet statistics, which is what both Phikon
and UNI expect, at the native 512 x 512 (no 1024 resize).

FROZENSE_PFM_RANDOM=1 builds the same architectures with random weights and
no download, for offline tests only.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import SEDecoder

PATCH = 16
HUB = {"pfm_phikon": "owkin/phikon", "pfm_uni": "hf-hub:MahmoodLab/uni"}


def _random() -> bool:
    return os.environ.get("FROZENSE_PFM_RANDOM") == "1"


class _PhikonEncoder(nn.Module):
    """Phikon through `transformers` (a ViTModel)."""

    def __init__(self):
        super().__init__()
        from transformers import ViTConfig, ViTModel
        if _random():
            self.vit = ViTModel(ViTConfig(), add_pooling_layer=False)
        else:
            self.vit = ViTModel.from_pretrained(HUB["pfm_phikon"], add_pooling_layer=False)
        # transformers 4.x: vit.encoder.layer; transformers 5.x: vit.layers
        layers = getattr(self.vit, "layers", None)
        if layers is None:
            layers = self.vit.encoder.layer
        self.blocks = list(layers)
        self.dim = self.vit.config.hidden_size
        self.prefix = 1                                  # CLS token

    def embed(self, x):
        return self.vit.embeddings(x, interpolate_pos_encoding=True)

    @staticmethod
    def run_block(block, h):
        out = block(h)
        return out[0] if isinstance(out, (tuple, list)) else out

    def final_norm(self, h):
        return self.vit.layernorm(h)

    def norm_module(self):
        return self.vit.layernorm


class _TimmViTEncoder(nn.Module):
    """UNI through `timm`, as its model card loads it."""

    def __init__(self):
        super().__init__()
        import timm
        if _random():
            self.vit = timm.create_model("vit_large_patch16_224", pretrained=False,
                                         init_values=1e-5, dynamic_img_size=True)
        else:
            self.vit = timm.create_model(HUB["pfm_uni"], pretrained=True,
                                         init_values=1e-5, dynamic_img_size=True)
        self.blocks = list(self.vit.blocks)
        self.dim = self.vit.embed_dim
        self.prefix = self.vit.num_prefix_tokens

    def embed(self, x):
        v = self.vit
        x = v.patch_embed(x)
        x = v._pos_embed(x)
        x = v.patch_drop(x)
        return v.norm_pre(x)

    @staticmethod
    def run_block(block, h):
        return block(h)

    def final_norm(self, h):
        return self.vit.norm(h)

    def norm_module(self):
        return self.vit.norm


def _proj(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True))


class PathologyViTSeg(nn.Module):
    """A pathology ViT encoder with the FrozenSE-SAM decoder."""

    def __init__(self, cfg, target_size: int = 512):
        super().__init__()
        if cfg.arch not in HUB:
            raise ValueError(f"unknown pathology encoder {cfg.arch!r}")
        self.cfg = cfg
        self.encoder = _PhikonEncoder() if cfg.arch == "pfm_phikon" else _TimmViTEncoder()
        enc = self.encoder
        depth = len(enc.blocks)

        # Freeze everything, then release the last N blocks and the final norm.
        for p in enc.parameters():
            p.requires_grad = False
        n = max(0, int(cfg.unfreeze_last_n_blocks))
        self.first_trainable = depth - n
        for block in enc.blocks[self.first_trainable:]:
            for p in block.parameters():
                p.requires_grad = True
        if n > 0:
            for p in enc.norm_module().parameters():
                p.requires_grad = True

        self.taps = (depth // 3 - 1, 2 * depth // 3 - 1)  # shallow -> 256 px, deep -> 128 px
        d = enc.dim
        self.neck = nn.Sequential(
            _proj(d, 256),
            nn.ConvTranspose2d(256, 256, 2, stride=2, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        skip_channels: tuple[int, ...] = ()
        if cfg.use_skip_connections:
            self.skip_deep = _proj(d, 128)      # -> 128 x 128, decoder stage 1
            self.skip_shallow = _proj(d, 64)    # -> 256 x 256, decoder stage 2
            skip_channels = (128, 64)
        self.decoder = SEDecoder(256, cfg, target_size, skip_channels=skip_channels)

    def _grid(self, tokens, b, gh, gw):
        t = tokens[:, self.encoder.prefix:, :]
        return t.transpose(1, 2).reshape(b, -1, gh, gw)

    def forward(self, x: torch.Tensor):
        b, _, h, w = x.shape
        gh, gw = h // PATCH, w // PATCH
        enc = self.encoder
        kept: dict[int, torch.Tensor] = {}

        with torch.no_grad():                        # the frozen part
            t = enc.embed(x)
            for i, block in enumerate(enc.blocks[:self.first_trainable]):
                t = enc.run_block(block, t)
                if i in self.taps:
                    kept[i] = t
        if self.first_trainable < len(enc.blocks):  # the fine-tuned tail
            t = t.detach()
            for i in range(self.first_trainable, len(enc.blocks)):
                t = enc.run_block(enc.blocks[i], t)
                if i in self.taps:
                    kept[i] = t
            t = enc.final_norm(t)
        else:
            with torch.no_grad():
                t = enc.final_norm(t)

        feats = self.neck(self._grid(t, b, gh, gw))
        skips = None
        if self.cfg.use_skip_connections:
            shallow, deep = self.taps
            s1 = F.interpolate(self.skip_deep(self._grid(kept[deep], b, gh, gw)),
                               size=(h // 4, w // 4), mode="bilinear", align_corners=False)
            s2 = F.interpolate(self.skip_shallow(self._grid(kept[shallow], b, gh, gw)),
                               size=(h // 2, w // 2), mode="bilinear", align_corners=False)
            skips = [s1, s2]
        return self.decoder(feats, skips)
