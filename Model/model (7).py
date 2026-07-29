"""
======================================================================
DeepSet Model for CF-NODE

Reference model accompanying the paper:

"CF-NODE: Counterfactual Spatial Node Intervention for
Test-Time Adaptation Under Domain Shift in Diabetic
Retinopathy Grading"

----------------------------------------------------------------------

Purpose
-------
This file provides the reference DeepSet model used to develop
and evaluate CF-NODE.

Although CF-NODE is model-agnostic, this implementation
demonstrates one compatible architecture that produces the
required outputs for counterfactual spatial node
intervention.

----------------------------------------------------------------------

Model Outputs
-------------

When return_nodes=True, the model returns:

1. logits
   Shape: (B, C)

   Final classification logits.

2. node_features
   Shape: (B, N, D)

   Spatial feature embeddings extracted from the final
   convolutional feature map. Each spatial position is
   treated as a node. A node is an internal representation
   and is not assumed to correspond directly to a
   pathological structure.

----------------------------------------------------------------------

Architecture

Input Image
      │
      ▼
ResNet50 Backbone
      │
      ▼
Feature Map
      │
      ├──────────────► GeM Pool
      │
      ▼
Node Projection
      │
      ▼
Attention Pool
      │
      ▼
Feature Fusion
      │
      ▼
Residual Fusion Block
      │
      ▼
Classifier

----------------------------------------------------------------------

Note
----
This model serves as a complete reference implementation.

Users may replace the backbone or network architecture with
their own implementation, provided the model returns compatible
outputs for CF-NODE.

======================================================================
"""

import torch
import torch.nn as nn
import torchvision.models as models


# =========================================================
# GeM GLOBAL POOLING
#
# Learns a generalized pooling operation that produces a
# robust global image representation from the backbone
# feature map.
# =========================================================
class GeMPool2d(nn.Module):

    def __init__(
        self,
        p=3.0,
        eps=1e-6
    ):

        super().__init__()

        self.p = nn.Parameter(
            torch.ones(1) * p
        )

        self.eps = eps

    def forward(self, x):

        return torch.mean(

            x.clamp(min=self.eps).pow(self.p),

            dim=(-1, -2)

        ).pow(1.0 / self.p)


# =========================================================
# ATTENTION-BASED NODE POOLING
#
# Learns the contribution of each spatial node and aggregates
# them into a single node-level representation.
# =========================================================
class AttentionPool(nn.Module):

    def __init__(
        self,
        dim=128
    ):

        super().__init__()

        self.attn = nn.Sequential(

            nn.Linear(dim, 64),

            nn.GELU(),

            nn.Dropout(0.15),

            nn.Linear(64, 1)
        )

    def forward(
        self,
        nodes
    ):

        scores = self.attn(nodes)

        weights = torch.softmax(
            scores,
            dim=1
        )

        pooled = (
            weights * nodes
        ).sum(dim=1)

        return pooled


# =========================================================
# FEATURE FUSION REFINEMENT
#
# Refines the concatenated global and node features using a
# residual MLP block.
# improves feature interaction
# =========================================================
class FusionBlock(nn.Module):

    def __init__(
        self,
        dim=256
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Linear(dim, dim),

            nn.LayerNorm(dim),

            nn.GELU(),

            nn.Dropout(0.15),

            nn.Linear(dim, dim)
        )

        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x
    ):

        residual = x

        x = self.block(x)

        x = x + residual

        x = self.norm(x)

        return x


# =========================================================
# DEEPSET MODEL
#
# Reference architecture compatible with CF-NODE.
# =========================================================
class DeepSetModel(nn.Module):

    def __init__(
        self,
        num_classes=5
    ):

        super().__init__()

        self.num_classes = num_classes

        # =================================================
        # RESNET50
        # =================================================
        backbone = models.resnet50(

            weights=models.ResNet50_Weights.DEFAULT
        )

        self.features = nn.Sequential(
            *list(backbone.children())[:-2]
        )

        feat_dim = 2048

        # =================================================
        # GeM GLOBAL POOL
        # =================================================
        self.gem_pool = GeMPool2d()

        # =================================================
        # NODE PROJECTION
        # =================================================
        self.node_proj = nn.Sequential(

            nn.Linear(
                feat_dim,
                128
            ),

            nn.LayerNorm(128),

            nn.GELU(),

            nn.Dropout(0.15)
        )

        # =================================================
        # ATTENTION POOL
        # =================================================
        self.attention_pool = AttentionPool(
            dim=128
        )

        # =================================================
        # GLOBAL FEATURE PROJECTION
        # =================================================
        self.global_proj = nn.Sequential(

            nn.Linear(
                feat_dim,
                128
            ),

            nn.LayerNorm(128),

            nn.GELU(),

            nn.Dropout(0.15)
        )

        # =================================================
        # FEATURE FUSION
        # =================================================
        fused_dim = 256

        # =================================================
        # FUSION REFINEMENT
        # =================================================
        self.fusion_block = FusionBlock(
            dim=fused_dim
        )

        # =================================================
        # FEATURE DROPOUT
        # =================================================
        self.feature_dropout = nn.Dropout(
            0.25
        )

        # =================================================
        # CLASSIFIER
        #
        # improved long-training stability
        # =================================================
        self.classifier = nn.Sequential(

            nn.Linear(
                fused_dim,
                128
            ),

            nn.LayerNorm(128),

            nn.GELU(),

            nn.Dropout(0.25),

            nn.Linear(
                128,
                64
            ),

            nn.GELU(),

            nn.Dropout(0.15),

            nn.Linear(
                64,
                num_classes
            )
        )

        # =================================================
        # INIT
        # =================================================
        self._init_weights()

    # =====================================================
    # INIT
    # =====================================================
    def _init_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Linear):

                nn.init.xavier_normal_(
                    m.weight
                )

                if m.bias is not None:

                    nn.init.zeros_(
                        m.bias
                    )

    # =====================================================
    # FORWARD
    # =====================================================
    def forward(
        self,
        x,
        return_nodes=True
    ):

        # =================================================
        # CNN FEATURES
        # =================================================
        feat = self.features(x)

        # =================================================
        # GLOBAL FEATURES
        # =================================================
        global_feat = self.gem_pool(
            feat
        )

        global_feat = self.global_proj(
            global_feat
        )

        # =================================================
        # GRID NODES
        # =================================================
        nodes = feat.flatten(2).transpose(
            1,
            2
        )

        # =================================================
        # NODE EMBEDDINGS
        # =================================================
        nodes = self.node_proj(
            nodes
        )

        # =================================================
        # ATTENTION NODE POOL
        # =================================================
        pooled_nodes = self.attention_pool(
            nodes
        )

        # =================================================
        # FUSION
        # =================================================
        fused = torch.cat(

            [
                pooled_nodes,
                global_feat
            ],

            dim=1
        )

        # =================================================
        # FUSION REFINEMENT
        # =================================================
        fused = self.fusion_block(
            fused
        )

        # =================================================
        # DROPOUT
        # =================================================
        fused = self.feature_dropout(
            fused
        )

        # =================================================
        # CLASSIFICATION
        # =================================================
        logits = self.classifier(
            fused
        )

        # =================================================
        # IMPORTANT
        #
        # CF-NODE expects the model to return:
        #
        #   logits
        #   node_features
        #
        # Changing this output format requires corresponding
        # changes inside cfnode.py.
        # =================================================
        if return_nodes:

            return (

                logits,

                nodes
            )

        return logits


# =========================================================
# FACTORY
# =========================================================
def get_model(cfg):

    model = DeepSetModel(
        num_classes=cfg.num_classes
    )

    return model