# PODNet Web Demo

Interactive Gradio demo for `PODNet_Training_v13_merged_classes_run2.ipynb`
(XAI-driven anaemia severity stratification from conjunctival images).

**Predict tab** – upload any test-set conjunctiva image and get:
- predicted class (Non-Anemic / Mild-Moderate / Severe) with cascade probabilities (horizontal-flip TTA, same as the notebook's test evaluation)
- Grad-CAM overlay + heatmap for the predicted class (two-stage-aware, `backbone.stage4.conv2`)
- evidential uncertainty score `u = K / S` from the Dirichlet head, plus Stage A/B details, Hb estimate and colour-channel attention weights

**Results Dashboard tab** – test-set KPIs, per-class metrics, confusion matrix, training curves,
sample Grad-CAM and training configuration (from `final_summary_run2.json`).

## Run locally

```bash
cd webapp
pip install -r requirements.txt
python app.py            # http://localhost:7860
```

## Docker

```bash
docker build -t podnet-demo webapp
docker run -p 8080:8080 podnet-demo   # http://localhost:8080
```

Weights: `checkpoints/podnet_best.pth` (best Phase 2 checkpoint, run 2).
