
# CF-NODE

## Counterfactual Lesion Node Editing for Test-Time Adaptation Under Domain Shift in Diabetic Retinopathy Grading

PyTorch implementation of **CF-NODE**, a Test-Time Adaptation (TTA) framework for improving domain robustness under domain shift in diabetic retinopathy grading through counterfactual lesion node editing.

> **CF-NODE: Counterfactual Spatial Node Intervention for Test-Time Adaptation in Diabetic Retinopathy Grading**

<img width="2500" height="1406" alt="CF_Node_FLow (2)" src="https://github.com/user-attachments/assets/c5d90dfc-3d23-4e2b-888c-d04a242faee9" />

---

## Overview

CF-NODE is a training-free Test-Time Adaptation framework that improves the robustness of diabetic retinopathy grading models under domain shift by identifying important lesion representations, generating counterfactual feature interventions, and refining predictions using ordinal reasoning and source prior correction.

---

## Repository Structure

```text
CF-NODE/

├── cfnode/
│   ├── cfnode.py
│   ├── config.py
│   └── __init__.py
│
├── model/
│   ├── model.py
│   ├── train.py
│   ├── config.py
│   └── README.md
│
├── figures/
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Installation

```bash
git clone https://github.com/yourusername/CF-NODE.git

cd CF-NODE

pip install -r requirements.txt
```

---

## Train the Reference Model

The repository includes the DeepSet reference model used in our experiments.

```bash
python model/train.py
```

---

## Apply CF-NODE

CF-NODE expects the model to return

```python
(
    logits,
    node_features,
    ordinal_logits
)
```

Once the model follows this interface, CF-NODE can be integrated directly into the inference pipeline.

```python
from cfnode.cfnode import run_tta

predictions = run_tta(
    model,
    loader,
    cfg
)
```

---

## Reference Model

The `model/` directory contains the complete DeepSet reference implementation used in our experiments, including

- model architecture
- training configuration
- training script

The reference model can be used directly or adapted for other datasets.

---

## License

This project is released under the MIT License. See the `LICENSE` file for details.
