"""PODNet model, preprocessing and Grad-CAM, ported from PODNet_Training_v13 (merged classes, run 2)."""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_SIZE = 224
NUM_CLASSES = 3
SEVERITY_ORDER = ["Non-Anemic", "Mild-Moderate", "Severe"]
BACKBONE_OUT_CHANNELS = 256
ANEMIC_DECISION_THRESHOLD = 0.35


# ---------------------------------------------------------------------------
# Preprocessing (identical to the notebook)
# ---------------------------------------------------------------------------
def _border_color(img, edge_sample_size=10):
    h, w = img.shape[:2]
    border_pixels = np.concatenate([
        img[0:edge_sample_size, :].reshape(-1, 3),
        img[h - edge_sample_size:h, :].reshape(-1, 3),
        img[:, 0:edge_sample_size].reshape(-1, 3),
        img[:, w - edge_sample_size:w].reshape(-1, 3),
    ], axis=0)
    return np.median(border_pixels, axis=0)


def crop_to_foreground(img, padding=5):
    h, w = img.shape[:2]
    background_color = _border_color(img)
    diff = np.abs(img.astype(np.float32) - background_color.astype(np.float32)).sum(axis=2)
    mask = diff > 30
    if mask.sum() == 0:
        return img
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmin, rmax = max(0, rmin - padding), min(h, rmax + padding)
    cmin, cmax = max(0, cmin - padding), min(w, cmax + padding)
    return img[rmin:rmax, cmin:cmax]


def compute_photometric_stem(rgb_img):
    background_color = _border_color(rgb_img)
    rgb = rgb_img.astype(np.float32)
    diff = np.abs(rgb - background_color.astype(np.float32)).sum(axis=2)
    foreground_mask = diff > 30

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    eps = 1e-6
    od_red = -np.log10(np.clip(r, eps, 255) / 255.0)
    od_green = -np.log10(np.clip(g, eps, 255) / 255.0)
    lab_a = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB).astype(np.float32)[..., 1]
    hsv_s = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV).astype(np.float32)[..., 1]

    od_red = od_red * foreground_mask
    od_green = od_green * foreground_mask
    lab_a = lab_a * foreground_mask
    hsv_s = hsv_s * foreground_mask

    if foreground_mask.sum() > 0:
        od_red_max = od_red[foreground_mask].max() + eps
        od_green_max = od_green[foreground_mask].max() + eps
        lab_a_min = lab_a[foreground_mask].min()
        lab_a_max = lab_a[foreground_mask].max() + eps
    else:
        od_red_max = od_red.max() + eps
        od_green_max = od_green.max() + eps
        lab_a_min = lab_a.min()
        lab_a_max = lab_a.max() + eps

    od_red_n = np.clip(od_red / od_red_max, 0, 1) * foreground_mask
    od_green_n = np.clip(od_green / od_green_max, 0, 1) * foreground_mask
    lab_a_n = np.clip((lab_a - lab_a_min) / (lab_a_max - lab_a_min + eps), 0, 1) * foreground_mask
    hsv_s_n = (hsv_s / 255.0) * foreground_mask

    stem = np.stack([r / 255.0, g / 255.0, b / 255.0, od_red_n, od_green_n, lab_a_n, hsv_s_n], axis=0)
    return stem.astype(np.float32)


def preprocess(rgb_img):
    """uint8 RGB array -> (cropped 224x224 RGB array, 7x224x224 stem tensor)."""
    img = crop_to_foreground(rgb_img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img, torch.from_numpy(compute_photometric_stem(img))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ColourChannelAttention(nn.Module):
    def __init__(self, num_photometric_channels=4, num_classes=NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_photometric_channels, 8),
                nn.ReLU(inplace=True),
                nn.Linear(8, num_photometric_channels),
            ) for _ in range(num_classes)
        ])

    def forward(self, photometric_channels, class_logits_prev=None):
        pooled = photometric_channels.mean(dim=[2, 3])
        stacked = torch.stack([branch(pooled) for branch in self.branches], dim=1)
        if class_logits_prev is not None:
            class_probs = F.softmax(class_logits_prev, dim=1).unsqueeze(-1)
        else:
            class_probs = torch.full(
                (photometric_channels.size(0), self.num_classes, 1),
                1.0 / self.num_classes, device=photometric_channels.device,
            )
        attention_weights = F.softmax((stacked * class_probs).sum(dim=1), dim=1)
        reweighted = photometric_channels * attention_weights.unsqueeze(-1).unsqueeze(-1)
        return reweighted, attention_weights


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcitation(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        return self.relu(out + identity)


class PODNetBackbone(nn.Module):
    def __init__(self, in_channels=7):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = ResidualBlock(32, 64, stride=2, dropout=0.1)
        self.stage2 = ResidualBlock(64, 128, stride=2, dropout=0.15)
        self.stage3 = ResidualBlock(128, 256, stride=2, dropout=0.2)
        self.stage4 = ResidualBlock(256, 256, stride=1, dropout=0.2)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.stage4(x)


class RadialZonePooling(nn.Module):
    def __init__(self, num_zones=3, background_threshold=0.05):
        super().__init__()
        self.num_zones = num_zones
        self.background_threshold = background_threshold

    def forward(self, feat_map, rgb_input=None):
        B, C, H, W = feat_map.shape
        device = feat_map.device
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij",
        )
        radius = torch.sqrt(xx ** 2 + yy ** 2)
        radius = radius / radius.max()
        zone_bounds = torch.linspace(0, 1, self.num_zones + 1, device=device)

        if rgb_input is not None:
            rgb_small = F.interpolate(rgb_input, size=(H, W), mode="bilinear", align_corners=False)
            foreground_mask = (rgb_small.mean(dim=1, keepdim=True) > self.background_threshold).float()
        else:
            foreground_mask = torch.ones(B, 1, H, W, device=device)

        zone_features = []
        for i in range(self.num_zones):
            radial_mask = ((radius >= zone_bounds[i]) & (radius < zone_bounds[i + 1])).float()
            combined_mask = radial_mask.unsqueeze(0).unsqueeze(0) * foreground_mask
            zone_sum = (feat_map * combined_mask).sum(dim=[2, 3])
            zone_count = combined_mask.sum(dim=[2, 3]) + 1e-6
            zone_features.append(zone_sum / zone_count)
        return torch.cat(zone_features, dim=1)


class TwoStageHead(nn.Module):
    def __init__(self, in_features, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.stageA_fc = nn.Linear(hidden_dim, 1)
        self.stageB_fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.stageA_fc(h).squeeze(-1), self.stageB_fc(h).squeeze(-1)

    @staticmethod
    def combine_to_3class_probs(stageA_logit, stageB_logit):
        p_anemic = torch.sigmoid(stageA_logit)
        p_severe_given_anemic = torch.sigmoid(stageB_logit)
        return torch.stack([
            1 - p_anemic,
            p_anemic * (1 - p_severe_given_anemic),
            p_anemic * p_severe_given_anemic,
        ], dim=1)


class HbRegressionHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_features, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, x):
        return self.fc(x).squeeze(-1)


class EvidentialHead(nn.Module):
    def __init__(self, in_features, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return F.softplus(self.fc(x)) + 1.0

    @staticmethod
    def alpha_to_probs_and_uncertainty(alpha):
        S = alpha.sum(dim=1, keepdim=True)
        return alpha / S, alpha.shape[1] / S.squeeze(-1)


class PODNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, num_zones=3):
        super().__init__()
        self.cca = ColourChannelAttention(num_photometric_channels=4, num_classes=num_classes)
        self.backbone = PODNetBackbone(in_channels=7)
        self.pooling = RadialZonePooling(num_zones=num_zones)
        pooled_dim = BACKBONE_OUT_CHANNELS * num_zones
        self.two_stage_head = TwoStageHead(pooled_dim)
        self.hb_head = HbRegressionHead(pooled_dim)
        self.evidential_head = EvidentialHead(pooled_dim, num_classes)

    def forward(self, x, class_logits_prev=None):
        rgb = x[:, 0:3, :, :]
        reweighted_photometric, attn_weights = self.cca(x[:, 3:7, :, :], class_logits_prev)
        feat_map = self.backbone(torch.cat([rgb, reweighted_photometric], dim=1))
        pooled = self.pooling(feat_map, rgb_input=rgb)
        stageA_logit, stageB_logit = self.two_stage_head(pooled)
        return {
            "stageA_logit": stageA_logit,
            "stageB_logit": stageB_logit,
            "hb_pred": self.hb_head(pooled),
            "alpha": self.evidential_head(pooled),
            "attn_weights": attn_weights,
        }


def load_model(checkpoint_path, device="cpu"):
    model = PODNet()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Inference (matches the notebook's test evaluation: horizontal-flip TTA, argmax)
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model, stem):
    x = stem.unsqueeze(0)
    out = model(x)
    out_flip = model(torch.flip(x, dims=[3]))
    probs = (
        TwoStageHead.combine_to_3class_probs(out["stageA_logit"], out["stageB_logit"])
        + TwoStageHead.combine_to_3class_probs(out_flip["stageA_logit"], out_flip["stageB_logit"])
    ) / 2.0
    evid_probs, uncertainty = EvidentialHead.alpha_to_probs_and_uncertainty(out["alpha"])
    return {
        "probs": probs[0].numpy(),
        "pred_idx": int(probs[0].argmax()),
        "p_anemic": float(torch.sigmoid(out["stageA_logit"])[0]),
        "p_severe_given_anemic": float(torch.sigmoid(out["stageB_logit"])[0]),
        "uncertainty": float(uncertainty[0]),
        "evidential_probs": evid_probs[0].numpy(),
        "alpha": out["alpha"][0].numpy(),
        "hb_pred": float(out["hb_pred"][0]),
        "cca_weights": out["attn_weights"][0].numpy(),
    }


def grad_cam(model, stem, target_class_idx):
    """Two-stage-aware Grad-CAM on backbone.stage4.conv2, as in the notebook."""
    model.eval()
    store = {}

    def forward_hook(module, inp, out):
        store["act"] = out
        if out.requires_grad:
            out.register_hook(lambda g: store.__setitem__("grad", g))

    handle = model.backbone.stage4.conv2.register_forward_hook(forward_hook)
    try:
        x = stem.unsqueeze(0)
        with torch.no_grad():
            prior = model(x)
        prior_probs = TwoStageHead.combine_to_3class_probs(prior["stageA_logit"], prior["stageB_logit"])
        # softmax(log p) == p, so CCA receives exactly this class prior
        outputs = model(x, class_logits_prev=torch.log(prior_probs + 1e-8))
        probs = TwoStageHead.combine_to_3class_probs(outputs["stageA_logit"], outputs["stageB_logit"])
        model.zero_grad()
        torch.log(probs[0, target_class_idx] + 1e-8).backward()
    finally:
        handle.remove()

    acts, grads = store["act"][0].detach(), store["grad"][0]
    weights = grads.mean(dim=[1, 2])
    cam = F.relu((weights[:, None, None] * acts).sum(dim=0))
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cv2.resize(cam.numpy(), (IMG_SIZE, IMG_SIZE))


def overlay_cam(rgb_img, cam, alpha=0.5):
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.uint8((1 - alpha) * rgb_img + alpha * heat)
