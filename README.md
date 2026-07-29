Reference DeepSet Model

This directory contains the reference DeepSet model used during the development and evaluation of CF-NODE.

The purpose of this model is to demonstrate the feature representations required by CF-NODE. It is not a required architecture—any model that provides compatible outputs can be used.

⸻

Folder Contents

model/
├── config.py      # Training configuration
├── model.py       # DeepSet model implementation
├── train.py       # Reference training script
└── README.md

⸻

Model Architecture

The reference model consists of:

* ResNet50 backbone for feature extraction
* GeM Pooling for global image representation
* DeepSet-inspired node representation
* Attention-based node pooling
* Residual feature fusion
* Classification head

This architecture was designed for diabetic retinopathy grading under domain shift and was used to evaluate CF-NODE.

⸻

Model Outputs

When calling:

logits, node_features = model(images)

the model returns two outputs.

1. Classification Logits

Shape:
(B, C)
Example:
(8, 5)

Final classification logits for diabetic retinopathy grading.

⸻

2. Node Features

Shape:
(B, N, D)
Example:
(8, 100, 128)

Feature embeddings extracted from the final feature map.

Each spatial location is treated as a node. A node is an internal representation and is not assumed to correspond directly to a pathological structure.

These node features are the primary input used by CF-NODE for counterfactual spatial node intervention.

⸻

Compatibility with CF-NODE

CF-NODE expects the model to return:

(
    logits,
    node_features
)

If your own model follows the same output format, it can be used with CF-NODE without modifying the algorithm.

⸻

Training

The provided train.py script demonstrates one training strategy for this reference model.

It includes:

* Progressive backbone fine-tuning
* Mixed Precision (AMP)
* Exponential Moving Average (EMA)
* Cross Entropy + Focal Loss
* Cosine Annealing Warm Restarts
* Early stopping

Researchers are free to train their own models using any optimization strategy.

CF-NODE is independent of the training procedure.

⸻

Using Your Own Model

You are not required to use this DeepSet architecture.

You may replace it with any model, including:

* ResNet
* EfficientNet
* ConvNeXt
* Vision Transformer (ViT)
* Swin Transformer
* DenseNet
* Custom architectures

As long as the model returns compatible outputs, CF-NODE can be integrated without changing the core algorithm.

⸻

Notes

This model is provided as a reference implementation to help researchers understand the expected feature representations used by CF-NODE.

The core contribution of this repository is the CF-NODE algorithm, while this model serves as an example of one compatible implementation.