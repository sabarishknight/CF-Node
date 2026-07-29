"""
======================================================================
Example DeepSet Model Configuration
This file contains the configuration used to train the example
DeepSet model included with this repository.
The trained model is fully compatible with CF-NODE and serves
as a reference implementation.
Modify this configuration when training your own model.
======================================================================
"""
import torch
class ModelConfig:
    # ==========================================================
    # DATASET
    # ==========================================================
    # Path to the training labels
    labels_path = "path/to/trainLabels.csv"
    # Directory containing training images
    img_dir = "path/to/images"
    # ==========================================================
    # OPTIONAL BALANCED SUBSET
    # ==========================================================
    subset_csv = None
    split_csv = None
    # ==========================================================
    # IMAGE SETTINGS
    # ==========================================================
    img_size = 320
    num_classes = 5
    # ==========================================================
    # DATALOADER
    # ==========================================================
    batch_size = 16
    num_workers = 2
    pin_memory = True
    persistent_workers = True
    prefetch_factor = 2
    # ==========================================================
    # TRAINING
    # ==========================================================
    epochs = 40
    lr = 1.5e-4
    weight_decay = 8e-5
    label_smoothing = 0.05
    use_amp = True
    grad_clip = 1.0
    # ==========================================================
    # DEVICE
    # ==========================================================
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_multi_gpu = torch.cuda.device_count() > 1
    cudnn_benchmark = False
    # ==========================================================
    # RANDOM SEED
    # ==========================================================
    seed = 123
    # ==========================================================
    # CHECKPOINT
    # ==========================================================
    save_path = "checkpoints/best_model.pth"
    