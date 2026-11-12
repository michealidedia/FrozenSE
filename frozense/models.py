"""FrozenSE models, U-Net baselines and zero-shot MobileSAM.

Decoder layout:
    encoder output      : 256 channels at 64 x 64
    stage 1  ConvT 2x2  : 256 -> 192, 64  -> 128,  then 2 x (3x3 conv + BN + ReLU)
    stage 2  ConvT 2x2  : 192 -> 96,  128 -> 256,  then 2 x (3x3 conv + BN + ReLU)
    stage 3  ConvT 2x2  : 96  -> 48,  256 -> 512,  then 2 x (3x3 conv + BN + ReLU)
    head     1x1 conv   : 48  -> 1
    output interpolated to the 512 x 512 patch grid
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
class SEBlock(nn.Module):
    """Squeeze-and-Excitation (Hu et al., 2018).
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = F.adaptive_avg_pool2d(x, 1).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpStage(nn.Module):
    """ConvTranspose -> conv blocks -> optional SE."""

    def __init__(self, in_ch: int, out_ch: int, n_convs: int = 2,
                 use_se: bool = True, se_reduction: int = 16,
                 skip_ch: int = 0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

        first_in = out_ch + skip_ch
        convs = [ConvBlock(first_in, out_ch)]
        convs += [ConvBlock(out_ch, out_ch) for _ in range(n_convs - 1)]
        self.convs = nn.Sequential(*convs)
        self.se = SEBlock(out_ch, se_reduction) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.act(self.norm(self.up(x)))
        if skip is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear",
                                     align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.convs(x)
        return self.se(x)


class SEDecoder(nn.Module):
    """The proposed decoder, with every design choice exposed as a switch."""

    def __init__(self, in_channels: int = 256, cfg: ModelConfig | None = None,
                 target_size: int = 512, skip_channels: Sequence[int] = ()):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.target_size = target_size
        self.aux_sdf = cfg.aux_sdf_head

        chans = cfg.decoder_channels
        stages = []
        prev = in_channels
        for i, ch in enumerate(chans):
            skip_ch = skip_channels[i] if i < len(skip_channels) else 0
            stages.append(UpStage(prev, ch, cfg.convs_per_stage,
                                  cfg.use_se, cfg.se_reduction, skip_ch=skip_ch))
            prev = ch
        self.stages = nn.ModuleList(stages)

        self.head = nn.Conv2d(prev, cfg.out_channels, kernel_size=1)
        self.sdf_head = nn.Conv2d(prev, 1, kernel_size=1) if cfg.aux_sdf_head else None

    def forward(self, feats: torch.Tensor, skips: Sequence[torch.Tensor] | None = None):
        x = feats
        for i, stage in enumerate(self.stages):
            skip = skips[i] if skips is not None and i < len(skips) else None
            x = stage(x, skip)

        logits = self.head(x)
        if logits.shape[-1] != self.target_size:
            logits = F.interpolate(logits, size=(self.target_size, self.target_size),
                                   mode="bilinear", align_corners=False)

        if self.sdf_head is not None:
            sdf = torch.tanh(self.sdf_head(x))
            if sdf.shape[-1] != self.target_size:
                sdf = F.interpolate(sdf, size=(self.target_size, self.target_size),
                                    mode="bilinear", align_corners=False)
            return logits, sdf
        return logits, None


# --------------------------------------------------------------------------
# MobileSAM encoder wrapper
# --------------------------------------------------------------------------
def load_mobilesam_encoder(checkpoint: str, freeze: bool = True,
                           unfreeze_last_n: int = 0) -> tuple[nn.Module, int]:
    """Load MobileSAM's TinyViT image encoder.

    Returns (encoder, output_channels).
    """
    try:
        from mobile_sam import sam_model_registry
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "MobileSAM is not installed. In Colab run:\n"
            "    !pip install -q git+https://github.com/ChaoningZhang/MobileSAM.git"
        ) from exc

    sam = sam_model_registry["vit_t"](checkpoint=checkpoint)
    encoder = sam.image_encoder

    # TinyViT keeps its ImageNet classifier (`norm_head`, `head`: 321,640
    # parameters) even inside MobileSAM, where forward() never calls it --
    # the lines are commented out in mobile_sam/modeling/tiny_vit_sam.py.
    # Left in place, those dead weights would inflate every parameter count
    for attr in ("head", "norm_head"):
        if isinstance(getattr(encoder, attr, None), nn.Module):
            setattr(encoder, attr, nn.Identity())

    if freeze:
        for p in encoder.parameters():
            p.requires_grad = False
        if unfreeze_last_n > 0:
            blocks = _encoder_blocks(encoder)
            for blk in blocks[-unfreeze_last_n:]:
                for p in blk.parameters():
                    p.requires_grad = True
            # The neck projects the last block's output to the 256 channels
            # the decoder reads. Fine-tuning the last blocks while keeping the
            # projection on top of them frozen would be an odd half-measure,
            # so it is unfrozen with them.
            neck = getattr(encoder, "neck", None)
            if neck is not None:
                for p in neck.parameters():
                    p.requires_grad = True

    return encoder, 256   # MobileSAM's neck emits 256 channels at 64x64


class _Checkpointed(nn.Module):
    """Recompute a block's activations in the backward pass instead of
    storing them.

    Training the whole TinyViT at 1024 x 1024 needs about 3 GB of
    activations per image (measured, FP32) -- around 17 GB at batch 6, more
    than a 16 GB T4 holds. Checkpointing cuts that to a fraction for some
    extra compute, with IDENTICAL arithmetic, provided two things hold, both
    handled here and in engine.train_one_epoch:

    * use_reentrant=False. The reentrant mode TinyViT's own `use_checkpoint`
      flag uses gives a block NO weight gradients when its input does not
      require grad -- which is exactly the partially unfrozen case, whose
      last stage reads the output of frozen stages. That stage would have
      silently stopped training.
    * BatchNorm statistics are not updated during training. Otherwise the
      recomputation would update running statistics a second time each step.
    """

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, *args):
        from torch.utils.checkpoint import checkpoint
        return checkpoint(self.block, *args, use_reentrant=False)


def enable_activation_checkpointing(encoder: nn.Module) -> int:
    """Checkpoint every block of every TinyViT stage that has trainable
    weights. Returns the number of blocks wrapped."""
    wrapped = 0
    for layer in getattr(encoder, "layers", []):
        blocks = getattr(layer, "blocks", None)
        if blocks is None or not any(p.requires_grad for p in layer.parameters()):
            continue
        for i, blk in enumerate(blocks):
            if not isinstance(blk, _Checkpointed):
                blocks[i] = _Checkpointed(blk)
                wrapped += 1
    return wrapped


def freeze_batchnorm_statistics(module: nn.Module) -> None:
    """Put every BatchNorm layer in eval mode: running statistics stay as
    pretrained, while any trainable affine weights still learn."""
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def _encoder_blocks(encoder: nn.Module) -> list[nn.Module]:
    """Best-effort discovery of transformer blocks for partial unfreezing."""
    for attr in ("layers", "blocks", "stages"):
        mod = getattr(encoder, attr, None)
        if isinstance(mod, (nn.ModuleList, nn.Sequential)):
            return list(mod)
    return [m for m in encoder.children() if isinstance(m, nn.Module)]


# MobileSAM's TinyViT encoder has fixed positional embeddings for a
# 1024 x 1024 input. Patches arrive at their native 512 x 512, so SAM-based
# models upsample INSIDE forward(), on the GPU. Keeping the resize in the
# model rather than the data pipeline matters for three reasons: the masks
# stay at native resolution, which the micrometre-calibrated HD95/ASD
# require; the U-Net baselines see 512 x 512 as they naturally would, rather
# than a needlessly upsampled image at four times the cost; and the
# interpolation runs on the GPU instead of in the CPU data workers.
SAM_INPUT_SIZE = 1024


def _to_sam_input(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] != (SAM_INPUT_SIZE, SAM_INPUT_SIZE):
        x = F.interpolate(x, size=(SAM_INPUT_SIZE, SAM_INPUT_SIZE),
                          mode="bilinear", align_corners=False)
    return x


# Encoder taps for the skip-connection ablation. At a
# 1024 x 1024 input, MobileSAM's TinyViT produces 64 channels at 256 x 256
# after `patch_embed`, and 128 channels at 128 x 128 (as tokens) after
# `layers.0`. Those line up with the decoder's first two up-stages, which
# output 128 x 128 and 256 x 256. The third stage (512 x 512) has no encoder
# feature at that resolution, so it takes none.
#   (module name, channels, spatial side) -- in decoder-stage order
SKIP_TAPS = (("layers.0", 128, 128), ("patch_embed", 64, 256))


def _as_spatial(t: torch.Tensor, side: int) -> torch.Tensor:
    """TinyViT emits (B, H*W, C) tokens from its stages; the decoder wants
    (B, C, H, W). Tokens are in row-major order, so this is the inverse of
    the encoder's own flatten-and-transpose."""
    if t.ndim == 3:
        b, n, c = t.shape
        if n != side * side:
            raise RuntimeError(f"skip tap has {n} tokens, expected {side}x{side}")
        t = t.transpose(1, 2).reshape(b, c, side, side)
    return t


class FrozenSESAM(nn.Module):
    """FrozenSE-SAM: MobileSAM TinyViT encoder (largely frozen) + SE decoder.

    SAM's prompt encoder and mask decoder are not used; the encoder feeds a
    task-specific convolutional decoder directly.
    """

    def __init__(self, cfg: ModelConfig, target_size: int = 512):
        super().__init__()
        self.encoder, enc_ch = load_mobilesam_encoder(
            cfg.sam_checkpoint, cfg.freeze_encoder, cfg.unfreeze_last_n_blocks
        )
        self.cfg = cfg
        if any(p.requires_grad for p in self.encoder.parameters()):
            enable_activation_checkpointing(self.encoder)

        # Skip connections (ablation): capture intermediate encoder outputs
        # with forward hooks. Without this the `use_skip_connections` switch
        # existed only in the config -- the "skip on" ablation would have
        # trained a model identical to the baseline.
        self._taps: dict[str, torch.Tensor] = {}
        skip_channels: tuple[int, ...] = ()
        if cfg.use_skip_connections:
            modules = dict(self.encoder.named_modules())
            for name, _, _ in SKIP_TAPS:
                if name not in modules:
                    raise RuntimeError(f"MobileSAM encoder has no module {name!r} to tap")
                modules[name].register_forward_hook(self._make_tap(name))
            skip_channels = tuple(c for _, c, _ in SKIP_TAPS)
        self.decoder = SEDecoder(enc_ch, cfg, target_size, skip_channels=skip_channels)

    def _make_tap(self, name: str):
        def tap(module, inputs, output):
            self._taps[name] = output
        return tap

    def forward(self, x: torch.Tensor):
        x = _to_sam_input(x)
        self._taps.clear()
        if self.cfg.freeze_encoder and self.cfg.unfreeze_last_n_blocks == 0:
            with torch.no_grad():
                feats = self.encoder(x)
        else:
            feats = self.encoder(x)

        skips = None
        if self.cfg.use_skip_connections:
            skips = [_as_spatial(self._taps[name], side) for name, _, side in SKIP_TAPS]
            self._taps.clear()
        return self.decoder(feats, skips)


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
class SMPWrapper(nn.Module):
    """Adapts segmentation_models_pytorch models to this project's interface."""

    def __init__(self, net: nn.Module, target_size: int = 512):
        super().__init__()
        self.net = net
        self.target_size = target_size

    def forward(self, x: torch.Tensor):
        logits = self.net(x)
        if logits.shape[-1] != self.target_size:
            logits = F.interpolate(logits, size=(self.target_size, self.target_size),
                                   mode="bilinear", align_corners=False)
        return logits, None


def build_unet(cfg: ModelConfig, pretrained: bool, target_size: int = 512) -> nn.Module:
    """U-Net baseline.
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "segmentation_models_pytorch is required for the U-Net baselines:\n"
            "    !pip install -q segmentation_models_pytorch"
        ) from exc

    net = smp.Unet(
        encoder_name=cfg.pretrained_encoder_name,
        encoder_weights=cfg.pretrained_encoder_weights if pretrained else None,
        in_channels=3,
        classes=cfg.out_channels,
    )
    return SMPWrapper(net, target_size)


class MobileSAMNative(nn.Module):
    """Frozen MobileSAM with its OWN prompt-based mask decoder.

    Prompting strategy: a uniform grid of point prompts over the patch, with
    the highest-confidence mask retained per point and the union taken. This
    is the standard automatic-mask-generation approximation.
    """

    def __init__(self, cfg: ModelConfig, target_size: int = 512, grid: int = 4):
        super().__init__()
        from mobile_sam import sam_model_registry

        self.sam = sam_model_registry["vit_t"](checkpoint=cfg.sam_checkpoint)
        for p in self.sam.parameters():
            p.requires_grad = False
        self.sam.eval()
        self.target_size = target_size
        self.grid = grid

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        b = x.shape[0]
        device = x.device
        size = self.target_size
        feats = self.sam.image_encoder(_to_sam_input(x))

        # Uniform grid of foreground point prompts. SAM's prompt encoder
        # expects coordinates in the frame of its 1024 x 1024 input.
        step = SAM_INPUT_SIZE / (self.grid + 1)
        pts = torch.tensor(
            [[(j + 1) * step, (i + 1) * step]
             for i in range(self.grid) for j in range(self.grid)],
            device=device, dtype=torch.float32,
        )
        n_pts = pts.shape[0]

        out = torch.zeros(b, 1, size, size, device=device)
        for bi in range(b):
            coords = pts.unsqueeze(1)                       # (n_pts, 1, 2)
            labels = torch.ones(n_pts, 1, device=device)    # all foreground
            sparse, dense = self.sam.prompt_encoder(
                points=(coords, labels), boxes=None, masks=None
            )
            low_res, iou_pred = self.sam.mask_decoder(
                image_embeddings=feats[bi : bi + 1],
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            masks = F.interpolate(low_res, size=(size, size), mode="bilinear",
                                  align_corners=False)
            keep = iou_pred.squeeze(-1) > 0.5
            if keep.any():
                union = (masks[keep] > 0).any(dim=0, keepdim=True).float()
            else:
                union = torch.zeros(1, 1, size, size, device=device)
            out[bi] = union.squeeze(0)

        logits = (out - 0.5) * 10.0   # pseudo-logits for a consistent interface
        return logits, None


def build_model(cfg: ModelConfig, target_size: int = 512) -> nn.Module:
    if cfg.arch == "frozense_sam":
        return FrozenSESAM(cfg, target_size)
    if cfg.arch == "mobilesam_plain":
        import dataclasses
        return FrozenSESAM(dataclasses.replace(cfg, use_se=False), target_size)
    if cfg.arch == "unet_pretrained":
        return build_unet(cfg, pretrained=True, target_size=target_size)
    if cfg.arch == "unet_scratch":
        return build_unet(cfg, pretrained=False, target_size=target_size)
    if cfg.arch == "mobilesam_native":
        return MobileSAMNative(cfg, target_size)
    if cfg.arch in ("pfm_phikon", "pfm_uni"):
        from .pathology import PathologyViTSeg
        return PathologyViTSeg(cfg, target_size)
    raise ValueError(f"Unknown architecture {cfg.arch!r}")


# --------------------------------------------------------------------------
def count_parameters(model: nn.Module) -> dict[str, int]:
    """Trainable and total parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
