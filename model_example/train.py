"""
======================================================================
Training Script for the Reference DeepSet Model

This script trains the reference DeepSet model provided with
this repository.

The resulting checkpoint is fully compatible with CF-NODE and
can be used directly for Test-Time Adaptation.

----------------------------------------------------------------------

Training Pipeline

1. Load training and validation datasets
2. Build the DeepSet model
3. Progressive backbone fine-tuning
4. EMA parameter averaging
5. Mixed precision training (AMP)
6. Validation using QWK, F1, Accuracy and AUC
7. Save the best checkpoint

----------------------------------------------------------------------

Loss Function

Total Loss =

    0.70 × Cross Entropy

  + 0.30 × Focal Loss

----------------------------------------------------------------------

Output

The best-performing model checkpoint is automatically saved
to the configured checkpoint directory.

======================================================================
"""
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import random

from torch.amp import autocast, GradScaler

from tqdm import tqdm

import numpy as np

from sklearn.metrics import (

    cohen_kappa_score,

    accuracy_score,

    f1_score,

)

from config import ModelConfig as CFG
from dataset import get_loaders
from model import get_model


# =========================================================
# SEED
# =========================================================
def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# =========================================================
# CUDA OPTIMIZATION
# =========================================================
torch.backends.cudnn.benchmark = False

torch.backends.cuda.matmul.allow_tf32 = True

torch.backends.cudnn.allow_tf32 = True

torch.set_float32_matmul_precision(
    "high"
)


# =========================================================
# EXPONENTIAL MOVING AVERAGE (EMA)
#
# Maintains a smoothed version of the model parameters for
# more stable validation and checkpoint selection.
# =========================================================
class EMA:

    def __init__(
        self,
        model,
        decay=0.999
    ):

        self.decay = decay

        self.shadow = {}

        self.backup = {}

        model = (
            model.module
            if isinstance(
                model,
                torch.nn.DataParallel
            )
            else model
        )

        for name, param in model.named_parameters():

            if param.requires_grad:

                self.shadow[name] = (
                    param.data.clone()
                )

    # =====================================================
    # FIXED EMA UPDATE
    #
    # handles newly unfrozen params
    # =====================================================
    def update(
        self,
        model
    ):

        model = (
            model.module
            if isinstance(
                model,
                torch.nn.DataParallel
            )
            else model
        )

        for name, param in model.named_parameters():

            if not param.requires_grad:
                continue

            # =============================================
            # NEW PARAM AFTER UNFREEZE
            # =============================================
            if name not in self.shadow:

                self.shadow[name] = (
                    param.data.clone()
                )

            # =============================================
            # EMA UPDATE
            # =============================================
            new_average = (

                self.decay *

                self.shadow[name]

                +

                (1.0 - self.decay) *

                param.data
            )

            self.shadow[name] = (
                new_average.clone()
            )

    def apply_shadow(
        self,
        model
    ):

        model = (
            model.module
            if isinstance(
                model,
                torch.nn.DataParallel
            )
            else model
        )

        for name, param in model.named_parameters():

            if not param.requires_grad:
                continue

            if name not in self.shadow:
                continue

            self.backup[name] = (
                param.data.clone()
            )

            param.data = (
                self.shadow[name]
            )

    def restore(
        self,
        model
    ):

        model = (
            model.module
            if isinstance(
                model,
                torch.nn.DataParallel
            )
            else model
        )

        for name, param in model.named_parameters():

            if name not in self.backup:
                continue

            param.data = (
                self.backup[name]
            )

        self.backup = {}


# =========================================================
# FOCAL LOSS
#
# Improves learning on difficult and minority samples by
# down-weighting easy examples.
# =========================================================
class FocalLoss(torch.nn.Module):

    def __init__(
        self,
        gamma=1.8
    ):

        super().__init__()

        self.gamma = gamma

    def forward(
        self,
        logits,
        targets
    ):

        ce = F.cross_entropy(

            logits,

            targets,

            reduction="none"
        )

        pt = torch.exp(-ce)

        focal = (
            (1 - pt) ** self.gamma
        ) * ce

        return focal.mean()


# =========================================================
# EXPECTED GRADE PREDICTION
#
# Converts class probabilities into a discrete grade using
# the posterior expected grade.
# =========================================================
def expected_grade(logits):

    probs = F.softmax(
        logits,
        dim=1
    )

    grades = torch.arange(

        logits.size(1),

        device=logits.device

    ).float()

    pred = (
        probs * grades
    ).sum(dim=1)

    pred = torch.clamp(
        torch.round(pred),
        0,
        logits.size(1) - 1
    )

    return pred.long()


# =========================================================
# FREEZE BACKBONE
# =========================================================
def freeze_backbone(model):

    target_model = (

        model.module
        if isinstance(
            model,
            torch.nn.DataParallel
        )
        else model
    )

    for param in target_model.features.parameters():

        param.requires_grad = False

    print(
        "\n🧊 Backbone frozen"
    )


# =========================================================
# UNFREEZE LAYER4
# =========================================================
def unfreeze_layer4(model):

    target_model = (

        model.module
        if isinstance(
            model,
            torch.nn.DataParallel
        )
        else model
    )

    for name, param in target_model.features.named_parameters():

        if "7" in name:

            param.requires_grad = True

    print(
        "\n🔥 Layer4 unfrozen"
    )


# =========================================================
# FULL BACKBONE
# =========================================================
def unfreeze_full_backbone(model):

    target_model = (

        model.module
        if isinstance(
            model,
            torch.nn.DataParallel
        )
        else model
    )

    for param in target_model.features.parameters():

        param.requires_grad = True

    print(
        "\n🚀 Full backbone unfrozen"
    )


# =========================================================
# VALIDATION
# =========================================================
@torch.no_grad()
def validate(

    model,

    loader,

    device,

    cfg,

    ce_criterion,

    focal_criterion
):

    model.eval()

    preds = []

    labels_all = []

    probs_all = []

    total_loss = 0.0

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        with autocast(

            device_type="cuda",

            enabled=cfg.use_amp
        ):

            logits, _ = model(
                images
            )

            ce_loss = ce_criterion(
                logits,
                labels
            )

            focal_loss = focal_criterion(
                logits,
                labels
            )

            loss = (

                0.70 * ce_loss

                +

                0.30 * focal_loss
            )

        total_loss += loss.item()

        pred = expected_grade(
            logits
        )

        probs = F.softmax(
            logits,
            dim=1
        )

        preds.extend(
            pred.cpu().numpy()
        )

        labels_all.extend(
            labels.cpu().numpy()
        )

        probs_all.extend(
            probs.cpu().numpy()
        )

    preds = np.array(preds)

    labels_all = np.array(labels_all)

    probs_all = np.array(probs_all)

    acc = accuracy_score(
        labels_all,
        preds
    )

    f1 = f1_score(
        labels_all,
        preds,
        average="macro"
    )

    qwk = cohen_kappa_score(
        labels_all,
        preds,
        weights="quadratic"
    )

    val_loss = (
        total_loss / len(loader)
    )

    return (

        val_loss,

        acc,

        f1,

        qwk
    )


# =========================================================
# MAIN
# =========================================================
def main():

    cfg = CFG()

    set_seed(cfg.seed)

    print(
        f"\n🎲 Seed: {cfg.seed}"
    )

    device = cfg.device

    print(
        f"💾 Save Path: {cfg.save_path}"
    )

    gpu_count = torch.cuda.device_count()

    if torch.cuda.is_available():

        print(
            f"🔥 GPUs: {gpu_count}"
        )

    # =====================================================
    # DATA
    # =====================================================
    train_loader, val_loader = get_loaders(cfg)

    # =====================================================
    # MODEL
    # =====================================================
    model = get_model(cfg)

    model = model.to(device)

    # =====================================================
    # MULTI GPU
    # =====================================================
    if gpu_count > 1:

        print(
            f"\n🚀 Using {gpu_count} GPUs"
        )

        model = torch.nn.DataParallel(
            model
        )

    # =====================================================
    # FREEZE STAGE
    # =====================================================
    freeze_backbone(model)

    # =====================================================
    # EMA
    # =====================================================
    ema = EMA(
        model,
        decay=0.999
    )

    # =====================================================
    # LOSS
    # =====================================================
    ce_criterion = torch.nn.CrossEntropyLoss(

        label_smoothing=cfg.label_smoothing
    )

    focal_criterion = FocalLoss(
        gamma=1.8
    )

    # =====================================================
    # OPTIMIZER
    # =====================================================
    optimizer = optim.AdamW(

        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),

        lr=cfg.lr,

        weight_decay=cfg.weight_decay,

        fused=True
    )

    # =====================================================
    # SCHEDULER
    # =====================================================
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(

            optimizer,

            T_0=10,

            T_mult=2,

            eta_min=1e-6
        )
    )

    # =====================================================
    # AMP
    # =====================================================
    scaler = GradScaler(

        "cuda",

        enabled=cfg.use_amp
    )

    # =====================================================
    # BEST
    # =====================================================
    best_score = -1.0

    best_qwk = -1.0

    best_f1 = -1.0


    # =====================================================
    # PATIENCE
    # =====================================================
    patience = 15

    patience_counter = 0

    print(
        "\n🔥 Improved Long Training Started\n"
    )

    # =====================================================
    # EPOCHS
    # =====================================================
    for epoch in range(cfg.epochs):

        # =================================================
        # STAGE 2
        # =================================================
        if epoch == 5:

            unfreeze_layer4(model)

            optimizer = optim.AdamW(

                filter(
                    lambda p: p.requires_grad,
                    model.parameters()
                ),

                lr=8e-5,

                weight_decay=cfg.weight_decay,

                fused=True
            )

            print(
                "\n🚀 Stage 2 Fine-tuning"
            )

        # =================================================
        # STAGE 3
        # =================================================
        if epoch == 15:

            unfreeze_full_backbone(model)

            optimizer = optim.AdamW(

                model.parameters(),

                lr=2e-5,

                weight_decay=cfg.weight_decay,

                fused=True
            )

            print(
                "\n🔥 Full Fine-tuning Stage"
            )

        model.train()

        running_loss = 0.0

        loop = tqdm(

            train_loader,

            leave=False
        )

        for images, labels in loop:

            images = images.to(

                device,

                non_blocking=True
            )

            labels = labels.to(

                device,

                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with autocast(

                device_type="cuda",

                enabled=cfg.use_amp
            ):

                logits, _ = model(
                    images
                )

                ce_loss = ce_criterion(
                    logits,
                    labels
                )

                focal_loss = focal_criterion(
                    logits,
                    labels
                )

                loss = (

                    0.70 * ce_loss

                    +

                    0.30 * focal_loss
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                cfg.grad_clip
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            # =============================================
            # EMA UPDATE
            # =============================================
            ema.update(model)

            running_loss += loss.item()

            loop.set_postfix(

                loss=f"{loss.item():.4f}"
            )

        scheduler.step(epoch + 1)

        # =================================================
        # EMA VALIDATION
        # =================================================
        ema.apply_shadow(model)

        (
            val_loss,
            acc,
            f1,
            qwk

        ) = validate(

            model,

            val_loader,

            device,

            cfg,

            ce_criterion,

            focal_criterion
        )

        ema.restore(model)

        train_loss = (
            running_loss / len(train_loader)
        )

        # =================================================
        # FINAL SCORE
        # =================================================
        score = (

            0.50 * f1

            +

            0.50 * qwk
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(

            f"Epoch {epoch+1:02d} | "

            f"LR {current_lr:.6f} | "

            f"Train {train_loss:.4f} | "

            f"Val {val_loss:.4f} | "

            f"Acc {acc:.4f} | "

            f"F1 {f1:.4f} | "

            f"QWK {qwk:.4f} | "
        )

        # =================================================
        # SAVE BEST
        # =================================================
        if score > best_score:

            best_score = score

            best_qwk = qwk

            best_f1 = f1

            best_auc = auc

            patience_counter = 0

            save_model = (

                model.module

                if isinstance(
                    model,
                    torch.nn.DataParallel
                )

                else model
            )

            torch.save(
            {
                "seed": cfg.seed,
                "model": save_model.state_dict(),
                "best_score": best_score,
                "best_f1": best_f1,
                "best_qwk": best_qwk,
            },
            cfg.save_path
            )

            print(

                f"✅ Saved Best | "

                f"Score {score:.4f} | "

                f"F1 {f1:.4f} | "

                f"QWK {qwk:.4f}"
            )

        else:

            patience_counter += 1

            print(

                f"⏳ Patience "

                f"{patience_counter}/{patience}"
            )

        # =================================================
        # EARLY STOPPING
        # =================================================
        if patience_counter >= patience:

            print(
                "\n🛑 Early stopping"
            )

            break

    # =====================================================
    # FINAL
    # =====================================================
    print(

    "\n🏁 Training Completed"

    )
    
    print(
    
        f"🎲 Seed Used: {cfg.seed}"
    
    )
    print(
        f"🏆 Best F1  : {best_f1:.4f}"
    )

    print(
        f"🏆 Best QWK : {best_qwk:.4f}"
    )


# =========================================================
# ENTRY
# =========================================================
if __name__ == "__main__":

    main()



"""
note:

This training script is provided as a reference implementation
for reproducing the DeepSet model used with CF-NODE.

Researchers are encouraged to train their own architectures
using their preferred training pipeline.

CF-NODE itself is independent of this training strategy.
"""