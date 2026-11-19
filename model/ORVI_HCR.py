"""Attention-based multiple instance learning and hierarchical ATP regression.

Use the bin boundaries and image preprocessing associated with each checkpoint.
"""

import math
from collections.abc import Mapping

import torch
from torch import nn

from .layers import Attention, Conv2D


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_label_ranges(label_ranges, category):
    """Return ordered log10 intervals, accepting JSON lists or integer keys."""
    if label_ranges is None:
        raise ValueError("label_ranges is required; recover it from training metadata")
    if isinstance(label_ranges, Mapping):
        if set(map(str, label_ranges)) != {str(i) for i in range(category)}:
            raise ValueError("label_ranges keys must be consecutive classes 0..category-1")
        values = [label_ranges[i] if i in label_ranges else label_ranges[str(i)]
                  for i in range(category)]
    else:
        values = list(label_ranges)
    if len(values) != category:
        raise ValueError("label_ranges length must equal category")
    ranges = []
    for interval in values:
        if len(interval) != 2:
            raise ValueError("Each label range must contain two log10 boundaries")
        # Retain NumPy scalar precision when bin edges come from np.linspace.
        lo, hi = (float(v) if isinstance(v, str) else v for v in interval)
        if not math.isfinite(lo) or not math.isfinite(hi) or not lo < hi:
            raise ValueError("Bin boundaries must be finite and strictly increasing")
        if ranges and not math.isclose(lo, ranges[-1][1], rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError("label_ranges must be ordered contiguous log10 intervals")
        ranges.append((lo, hi))
    return ranges


def decode_hcr(logits, pcts, label_ranges, prob_threshold=0.25, fallback_topk=3):
    """Decode bin positions and mix ATP in LINEAR units, as in AD2DMIT.py.

    Only probabilities strictly above the threshold participate. If none pass,
    use the largest min(fallback_topk, K) probabilities. The recovered default
    is 3; other values are explicit experimental settings saved with the model.
    """
    if logits.ndim != 2 or logits.shape != pcts.shape or logits.shape[1] < 1:
        raise ValueError("logits and pcts must have the same [batch, classes] shape")
    if not 0 <= prob_threshold <= 1:
        raise ValueError("prob_threshold must lie in [0, 1]")
    _positive_int("fallback_topk", fallback_topk)
    ranges = torch.as_tensor(label_ranges, device=pcts.device)
    if ranges.shape != (logits.shape[1], 2):
        raise ValueError("label_ranges must have shape [classes, 2]")
    probs = logits.softmax(dim=1)
    lower = ranges[:, 0]
    widths = ranges[:, 1] - lower
    bin_log10 = lower + pcts * widths
    bin_atp = torch.pow(10.0, bin_log10)
    # The original CSV predictor reduces only the selected bins, in index order
    # above threshold and in top-k order on fallback. Its ATP output uses the
    # default floating dtype even when the bin boundaries use higher precision.
    atp = torch.empty((len(probs), 1), device=probs.device, dtype=torch.get_default_dtype())
    weights = torch.zeros_like(probs)
    for i, row in enumerate(probs):
        selected = torch.where(row > prob_threshold)[0]
        if selected.numel() == 0:
            selected = row.topk(min(fallback_topk, row.numel())).indices
        selected_probs = row[selected]
        normalized = selected_probs / selected_probs.sum()
        weights[i, selected] = normalized
        atp[i, 0] = (bin_atp[i, selected] * normalized).sum()
    winner = probs.argmax(dim=1)
    position = pcts.gather(1, winner.unsqueeze(1)).flatten()
    argmax_atp = torch.pow(10.0, lower[winner] + widths[winner] * position).unsqueeze(1)
    return {"atp": atp, "bin_atp": bin_atp, "bin_log10": bin_log10,
            "argmax_atp": argmax_atp, "bin_probs": probs, "mixture_weights": weights}


class ORVIClassifier(nn.Module):
    """Phase 1 classifier with the exact legacy feature/attention/head layout.

    ``patches`` historically controls depth via floor(log2(patches))//2. It is
    equal to the actual bag size only for certain input sizes, including 512
    with patches=64. ``num_layer`` was ignored in the original implementation
    and remains an accepted compatibility argument. The depth is derived.
    """

    def __init__(self, in_channel=1, hidden=256, category=9, num_layer=None,
                 image_size=512, patches=64, visualize=False):
        super().__init__()
        for name, value in (("in_channel", in_channel), ("hidden", hidden),
                            ("category", category), ("image_size", image_size),
                            ("patches", patches)):
            _positive_int(name, value)
        temp = int(math.log2(patches)) // 2
        depth = 9 - temp
        if depth < 1 or image_size < 2 ** depth:
            raise ValueError(f"patches={patches} gives {depth} pooling stages; "
                             f"image_size must be at least {2 ** max(depth, 0)}")
        self.in_channel, self.hidden, self.category = in_channel, hidden, category
        self.image_size, self.patches, self.num_layer = image_size, patches, depth
        self.visualize = visualize
        self._frozen_modules = set()
        channels = in_channel
        feature_extractor = []
        factor = 2 ** (temp + 2)
        for i in range(depth):
            width = factor * (2 ** i)
            feature_extractor.extend([Conv2D(channels, width, ac=True), nn.MaxPool2d(2, 2)])
            channels = width
        feature_extractor.append(Conv2D(channels, hidden, ac=True))
        self.feature_extractor = nn.Sequential(*feature_extractor)
        self.attention = Attention(hidden, hidden * 2, (image_size // (2 ** depth)) ** 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.BatchNorm1d(hidden * 2), nn.ReLU(),
            nn.Linear(hidden * 2, hidden * 4), nn.BatchNorm1d(hidden * 4), nn.ReLU(),
            nn.Linear(hidden * 4, category),
        )

    def get_config(self):
        return {"in_channel": self.in_channel, "hidden": self.hidden,
                "category": self.category, "image_size": self.image_size,
                "patches": self.patches}

    def to_config(self):
        return self.get_config()

    def freeze_representation(self, freeze_classifier=False):
        """Freeze weights AND BatchNorm statistics for a regression-only phase."""
        names = {"feature_extractor", "attention"}
        if freeze_classifier:
            names.add("classifier")
        for name in names:
            module = getattr(self, name)
            module.requires_grad_(False)
            module.eval()
        self._frozen_modules.update(names)
        return self

    def unfreeze_all(self):
        self._frozen_modules.clear()
        self.requires_grad_(True)
        self.train(self.training)
        return self

    def train(self, mode=True):
        super().train(mode)
        for name in self._frozen_modules:
            getattr(self, name).eval()
        return self

    def _encode(self, x):
        expected = (self.in_channel, self.image_size, self.image_size)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected or x.shape[0] < 1:
            raise ValueError(f"Expected a nonempty input [B, {expected[0]}, "
                             f"{expected[1]}, {expected[2]}], got {tuple(x.shape)}")
        features = self.feature_extractor(x)
        instances = features.permute(0, 2, 3, 1).contiguous().view(-1, self.hidden)
        weights = self.attention(instances)
        instances = instances.view(x.shape[0], -1, self.hidden)
        pooled = (instances * weights.unsqueeze(-1)).sum(dim=1)
        spatial_attention = weights.view(x.shape[0], features.shape[2], features.shape[3])
        return pooled, features, spatial_attention

    def forward_details(self, x):
        pooled, features, attention = self._encode(x)
        return {"logits": self.classifier(pooled), "features": features,
                "attention": attention, "class_attention": attention}

    def forward(self, x):
        return self.forward_details(x)["logits"]

    def forward_feat(self, x):
        details = self.forward_details(x)
        return details["logits"], details["features"]


class ORVIHCR(ORVIClassifier):
    """CNN/MIL encoder, interval classifier, and one sigmoid regressor per bin.

    Attention is shared across classes: it is NOT a class-specific explanation.
    ``class_attention`` is a compatibility alias of that same spatial map.
    """

    def __init__(self, in_channel=1, hidden=256, category=9, image_size=512,
                 label_ranges=None, patches=64, visualize=False,
                 prob_threshold=0.25, fallback_topk=3, range_dtype=None):
        ranges = validate_label_ranges(label_ranges, category)
        if not 0 <= prob_threshold <= 1:
            raise ValueError("prob_threshold must lie in [0, 1]")
        _positive_int("fallback_topk", fallback_topk)
        super().__init__(in_channel, hidden, category, image_size=image_size,
                         patches=patches, visualize=visualize)
        self.label_ranges = {i: tuple(map(float, pair)) for i, pair in enumerate(ranges)}
        # Keep interval metadata at its original precision across .half() or
        # .double() calls. decode_hcr transfers it to the input device as needed.
        dtype = None if range_dtype is None else getattr(torch, range_dtype, None)
        if range_dtype is not None and not isinstance(dtype, torch.dtype):
            raise ValueError("range_dtype must name a torch dtype")
        self._ranges = torch.tensor(ranges, dtype=dtype)
        self.prob_threshold, self.fallback_topk = prob_threshold, fallback_topk
        self.reg_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden * 2), nn.BatchNorm1d(hidden * 2), nn.ReLU(),
                nn.Linear(hidden * 2, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                nn.Linear(hidden, 1), nn.Sigmoid(),
            ) for _ in range(category)
        ])

    def get_config(self):
        config = super().get_config()
        config.update(label_ranges=[list(self.label_ranges[i]) for i in range(self.category)],
                      range_dtype=str(self._ranges.dtype).split(".")[-1],
                      prob_threshold=self.prob_threshold, fallback_topk=self.fallback_topk)
        return config

    def forward_details(self, x):
        pooled, features, attention = self._encode(x)
        logits = self.classifier(pooled)
        pcts = torch.cat([head(pooled) for head in self.reg_heads], dim=1)
        details = decode_hcr(logits, pcts, self._ranges, self.prob_threshold, self.fallback_topk)
        details.update(logits=logits, pcts=pcts, features=features,
                       attention=attention, class_attention=attention)
        return details

    def forward(self, x):
        details = self.forward_details(x)
        return details["atp"], details["logits"], details["pcts"], details["argmax_atp"]


class ORVIRegressor(ORVIClassifier):
    """Continuous baseline predicting log10(ATP) directly (no output activation)."""

    def __init__(self, in_channel=1, hidden=256, image_size=512, patches=64, visualize=False):
        super().__init__(in_channel, hidden, 1, image_size=image_size,
                         patches=patches, visualize=visualize)

    def get_config(self):
        config = super().get_config()
        config.pop("category")
        return config

    def forward(self, x):
        return super().forward(x).squeeze(-1)


class ORVISingleHead(ORVIClassifier):
    """Historical ablation: one within-bin position shared by all classes."""

    def __init__(self, in_channel=1, hidden=256, category=9, image_size=512,
                 patches=64, visualize=False):
        super().__init__(in_channel, hidden, category, image_size=image_size,
                         patches=patches, visualize=visualize)
        self.quantile_head = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.BatchNorm1d(hidden * 2), nn.ReLU(),
            nn.Linear(hidden * 2, hidden * 4), nn.BatchNorm1d(hidden * 4), nn.ReLU(),
            nn.Linear(hidden * 4, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        pooled, _, _ = self._encode(x)
        return self.classifier(pooled), self.quantile_head(pooled).view(-1)


# Compatibility spellings; all implementations live in this module.
ORVI_HCR = ORVIClassifier
ORVI_HCR_bin_MutiQ = ORVIHCR
ORVI_HCR_Log_Continue = ORVIRegressor
ORVI_HCR_bin_Q = ORVISingleHead
