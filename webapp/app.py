import json
import os
import time

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from modeling_podnet import (
    ANEMIC_DECISION_THRESHOLD,
    SEVERITY_ORDER,
    grad_cam,
    load_model,
    overlay_cam,
    predict,
    preprocess,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(HERE, "checkpoints", "podnet_best.pth")
SUMMARY_PATH = os.path.join(HERE, "final_summary_run2.json")
ASSETS = os.path.join(HERE, "assets")
DEVICE = "cpu"

print("Loading PODNet on CPU...")
model = load_model(CHECKPOINT_PATH, device=DEVICE)
num_params = sum(p.numel() for p in model.parameters())
print(f"Model loaded ({num_params:,} parameters).")

with open(SUMMARY_PATH) as f:
    SUMMARY = json.load(f)

CCA_CHANNELS = ["OD red", "OD green", "LAB a*", "HSV saturation"]


def uncertainty_band(u):
    if u < 0.35:
        return "Low", "#16a34a"
    if u < 0.65:
        return "Moderate", "#d97706"
    return "High", "#dc2626"


def bar(label, value, color):
    pct = max(0.0, min(1.0, value)) * 100
    return (
        f"<div style='margin:4px 0'><div style='display:flex;justify-content:space-between;font-size:13px'>"
        f"<span>{label}</span><span>{value:.3f}</span></div>"
        f"<div style='background:rgba(128,128,128,.2);border-radius:4px;height:8px'>"
        f"<div style='width:{pct:.1f}%;background:{color};height:8px;border-radius:4px'></div></div></div>"
    )


def analyse(image):
    if image is None:
        raise gr.Error("Please upload a conjunctiva image first.")

    rgb = np.array(image.convert("RGB"))
    cropped, stem = preprocess(rgb)

    start = time.perf_counter()
    res = predict(model, stem)
    infer_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    cam = grad_cam(model, stem, res["pred_idx"])
    cam_ms = (time.perf_counter() - start) * 1000

    pred_name = SEVERITY_ORDER[res["pred_idx"]]
    u = res["uncertainty"]
    level, color = uncertainty_band(u)

    label = {name: float(p) for name, p in zip(SEVERITY_ORDER, res["probs"])}

    uncertainty_html = (
        f"<div style='padding:14px;border-radius:10px;border:2px solid {color}'>"
        f"<div style='font-size:13px;opacity:.75'>Evidential uncertainty (u = K / S)</div>"
        f"<div style='font-size:34px;font-weight:700;color:{color}'>{u:.3f}</div>"
        f"<div style='font-weight:600;color:{color}'>{level} uncertainty</div>"
        f"<div style='font-size:12px;opacity:.75;margin-top:6px'>Dirichlet strength S = {res['alpha'].sum():.2f} "
        f"&middot; confidence 1 &minus; u = {1 - u:.3f}</div>"
        + "".join(bar(f"Evidence &alpha; {n}", a / res["alpha"].sum(), "#6366f1")
                  for n, a in zip(SEVERITY_ORDER, res["alpha"]))
        + "</div>"
    )

    stage_a = "Anemic" if res["p_anemic"] > ANEMIC_DECISION_THRESHOLD else "Non-Anemic"
    stage_b = "Severe" if res["p_severe_given_anemic"] > 0.5 else "Mild-Moderate"
    details_html = (
        "<div style='font-size:14px;line-height:1.5'>"
        f"<b>Stage A</b> (anemic vs not): P(anemic) = {res['p_anemic']:.3f} &rarr; {stage_a} "
        f"(threshold {ANEMIC_DECISION_THRESHOLD})<br>"
        f"<b>Stage B</b> (severe vs mild-moderate | anemic): P(severe) = {res['p_severe_given_anemic']:.3f} &rarr; {stage_b}<br>"
        f"<b>Estimated Hb</b>: {res['hb_pred']:.2f} g/dL "
        f"<span style='opacity:.7'>(test MAE &plusmn;{SUMMARY['hb_mae']:.2f})</span><br>"
        f"<b>Colour-channel attention</b>:"
        + "".join(bar(n, w, "#0ea5e9") for n, w in zip(CCA_CHANNELS, res["cca_weights"]))
        + f"<div style='font-size:12px;opacity:.7;margin-top:6px'>Inference (with flip TTA): {infer_ms:.0f} ms &middot; "
          f"Grad-CAM: {cam_ms:.0f} ms &middot; CPU</div></div>"
    )

    heat_only = overlay_cam(np.zeros_like(cropped), cam, alpha=1.0)
    return (
        label,
        f"## Predicted class: {pred_name}",
        uncertainty_html,
        cropped,
        overlay_cam(cropped, cam),
        heat_only,
        details_html,
    )


def adjusted_uncertainty(path, offset):
    if not path:
        return ""
    stem_name = os.path.splitext(os.path.basename(path))[0]
    _, stem = preprocess(np.array(Image.open(path).convert("RGB")))
    raw = predict(model, stem)["uncertainty"]
    sign = 1 if stem_name.endswith("_") else -1
    adjusted = float(np.clip(raw + sign * offset, 0.0, 1.0))
    level, color = uncertainty_band(adjusted)
    return (
        f"<div style='padding:14px;border-radius:10px;border:2px dashed {color}'>"
        f"<div style='font-size:12px;font-weight:700;color:#dc2626'>MANUALLY ADJUSTED, NOT MODEL OUTPUT</div>"
        f"<div style='font-size:13px;opacity:.75;margin-top:6px'>File: {stem_name} "
        f"&rarr; offset {'+' if sign > 0 else '&minus;'}{offset:.2f}</div>"
        f"<div style='font-size:34px;font-weight:700;color:{color}'>{adjusted:.3f}</div>"
        f"<div style='font-weight:600;color:{color}'>{level} uncertainty (adjusted)</div>"
        f"<div style='font-size:13px;margin-top:6px'>Raw model uncertainty: <b>{raw:.3f}</b></div></div>"
    )


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
def kpi(title, value, sub=""):
    return (
        "<div style='flex:1;min-width:150px;padding:14px;border-radius:10px;"
        "border:1px solid rgba(128,128,128,.3)'>"
        f"<div style='font-size:12px;opacity:.7'>{title}</div>"
        f"<div style='font-size:28px;font-weight:700'>{value}</div>"
        f"<div style='font-size:12px;opacity:.7'>{sub}</div></div>"
    )


def dashboard_kpis():
    s = SUMMARY
    tiles = [
        kpi("Test accuracy", f"{s['model_test_acc']:.1%}", f"naive baseline {s['naive_baseline_acc']:.1%}"),
        kpi("Macro F1", f"{s['test_classification_report']['macro avg']['f1-score']:.3f}", "3 classes"),
        kpi("Stage A accuracy", f"{s['stageA_binary_accuracy']:.1%}", "anemic vs non-anemic"),
        kpi("Stage B accuracy", f"{s['stageB_accuracy_among_correctly_passed']:.1%}", "severe vs mild-moderate"),
        kpi("Hb MAE", f"{s['hb_mae']:.2f} g/dL", "regression head"),
        kpi("ECE", f"{s['ece']:.3f}", "evidential confidence"),
        kpi("Parameters", f"{s['total_params'] / 1e6:.2f} M", "PODNet"),
        kpi("Anemic missed at Stage A", str(s["truly_anemic_lost_at_stageA"]), "test set"),
    ]
    return "<div style='display:flex;flex-wrap:wrap;gap:10px'>" + "".join(tiles) + "</div>"


def per_class_table():
    rep = SUMMARY["test_classification_report"]
    rows = []
    for name in SEVERITY_ORDER + ["macro avg", "weighted avg"]:
        r = rep[name]
        rows.append([name, round(r["precision"], 3), round(r["recall"], 3),
                     round(r["f1-score"], 3), int(r["support"])])
    return pd.DataFrame(rows, columns=["Class", "Precision", "Recall", "F1", "Support"])


def per_class_plot():
    rep = SUMMARY["test_classification_report"]
    metrics = ["precision", "recall", "f1-score"]
    colors = ["#6366f1", "#0ea5e9", "#10b981"]
    x = np.arange(len(SEVERITY_ORDER))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, (m, c) in enumerate(zip(metrics, colors)):
        vals = [rep[n][m] for n in SEVERITY_ORDER]
        bars = ax.bar(x + (i - 1) * 0.26, vals, 0.26, label=m.replace("-score", ""), color=c)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x, [f"{n}\n(n={int(rep[n]['support'])})" for n in SEVERITY_ORDER])
    ax.set_ylim(0, 1.12)
    ax.axhline(SUMMARY["naive_baseline_acc"], ls="--", color="gray", lw=1)
    ax.text(1.6, SUMMARY["naive_baseline_acc"] + 0.02, "naive baseline acc", ha="center", fontsize=8, color="gray")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=9)
    ax.set_title("Per-class test metrics (with flip TTA)")
    fig.tight_layout()
    return fig


def config_table():
    s = SUMMARY
    hp, td = s["phase1_hyperparameters_used"], s["training_design"]
    rows = [
        ["Classes", "Non-Anemic / Mild-Moderate / Severe (Mild + Moderate merged)"],
        ["Input", "224x224, 7-channel photometric stem (RGB, OD red/green, LAB a*, HSV S)"],
        ["Backbone", "Residual CNN with SE blocks (32 -> 256 ch) + Colour-Channel Attention"],
        ["Pooling", "Background-aware radial-zone pooling (3 zones)"],
        ["Heads", "Two-stage cascade, Hb regression, evidential (Dirichlet) uncertainty"],
        ["Phase 1", f"LR {hp['LR']}, batch {hp['BATCH_SIZE']}, WD {hp['WEIGHT_DECAY']}, "
                    f"{td['phase1_epochs_planned']} epochs, patience {td['phase1_patience']}"],
        ["Phase 2", f"Hb + evidential heads, backbone LR {td['phase2_backbone_lr']:.0e}, "
                    f"{td['phase2_epochs_planned']} epochs, patience {td['phase2_patience']}"],
        ["Stage A threshold", str(s["anemic_decision_threshold"])],
        ["Best Phase 1 smoothed val acc", f"{s['best_phase1_smoothed_val_acc']:.3f}"],
        ["Best Phase 2 val loss", f"{s['best_phase2_val_loss']:.3f}"],
    ]
    return pd.DataFrame(rows, columns=["Setting", "Value"])


with gr.Blocks(title="PODNet | XAI Anaemia Severity") as demo:
    gr.Markdown(
        """
        # PODNet | Explainable Anaemia Severity from Conjunctival Images

        Upload a palpebral conjunctiva image (e.g. from the test set). PODNet predicts the anaemia
        severity class, shows a **Grad-CAM** map of the regions that drove the decision, and reports an
        **evidential uncertainty** score from its Dirichlet head.
        """
    )

    with gr.Tab("Predict"):
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Conjunctiva image", height=360)
                with gr.Row():
                    submit_btn = gr.Button("Analyse", variant="primary")
                    clear_btn = gr.Button("Clear")
                preproc_image = gr.Image(label="Model input (foreground-cropped, 224x224)",
                                         interactive=False, height=260)
            with gr.Column(scale=1):
                pred_md = gr.Markdown("## Predicted class: -")
                label_out = gr.Label(label="Class probabilities (two-stage cascade, flip TTA)", num_top_classes=3)
                uncertainty_out = gr.HTML()
        with gr.Row():
            cam_overlay = gr.Image(label="Grad-CAM overlay (predicted class)", interactive=False, height=300)
            cam_heat = gr.Image(label="Grad-CAM heatmap", interactive=False, height=300)
            details_out = gr.HTML()

        gr.Markdown(
            "**Pipeline:** image -> border-colour-aware crop -> photometric stem -> "
            "colour-channel attention -> SE-ResNet backbone -> radial-zone pooling -> "
            "Stage A / Stage B cascade + Hb + evidential heads"
        )

        outputs = [label_out, pred_md, uncertainty_out, preproc_image, cam_overlay, cam_heat, details_out]
        submit_btn.click(analyse, inputs=input_image, outputs=outputs)
        input_image.upload(analyse, inputs=input_image, outputs=outputs)
        clear_btn.click(
            lambda: (None, None, "## Predicted class: -", "", None, None, None, ""),
            outputs=[input_image] + outputs,
        )

    with gr.Tab("Results Dashboard"):
        gr.Markdown("### Held-out test set results (run 2, 128 images)")
        gr.HTML(dashboard_kpis())
        with gr.Row():
            with gr.Column(scale=3):
                gr.Plot(per_class_plot(), label="Per-class metrics")
            with gr.Column(scale=2):
                gr.Dataframe(per_class_table(), label="Classification report", interactive=False)
        with gr.Row():
            gr.Image(os.path.join(ASSETS, "confusion_matrix.png"), label="Confusion matrix (with TTA)",
                     interactive=False)
            gr.Image(os.path.join(ASSETS, "gradcam_sample.png"), label="Per-class Grad-CAM (test sample)",
                     interactive=False)
        gr.Image(os.path.join(ASSETS, "training_curves.png"), label="Training curves (Phase 1 & Phase 2)",
                 interactive=False)
        gr.Dataframe(config_table(), label="Model & training configuration", interactive=False)
        gr.Markdown(
            "_Note: Severe has only 7 test images, so its recall (0.43) is a high-variance estimate. "
            "ECE is computed from the evidential confidence (1 - u)._"
        )

        gr.Markdown(
            "### Uncertainty adjustment demo (manual, not model output)\n"
            "This section applies a **manual offset** to the model's evidential uncertainty. "
            "Rule: if the filename (without extension) ends in `_`, the offset is **added**; "
            "otherwise it is **subtracted**. The raw model value is always shown alongside."
        )
        with gr.Row():
            with gr.Column(scale=1):
                adj_image = gr.Image(type="filepath", label="Image", height=280)
                adj_slider = gr.Slider(0.0, 0.5, value=0.2, step=0.01,
                                       label="Manual uncertainty offset (demo adjustment)")
            with gr.Column(scale=1):
                adj_out = gr.HTML()

        adj_inputs = [adj_image, adj_slider]
        adj_image.change(adjusted_uncertainty, inputs=adj_inputs, outputs=adj_out)
        adj_slider.release(adjusted_uncertainty, inputs=adj_inputs, outputs=adj_out)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=True,
        theme=gr.themes.Soft(),
    )
