"""
======================================================================
CF-NODE
Counterfactual Spatial Node Intervention for Test-Time Adaptation

Official implementation accompanying the paper:

"CF-NODE: Counterfactual Spatial Node Intervention for
Test-Time Adaptation Under Domain Shift in Diabetic
Retinopathy Grading"

----------------------------------------------------------------------

Purpose
-------
This module implements the complete CF-NODE algorithm for
source-free and label-free Test-Time Adaptation (TTA).

CF-NODE improves the robustness of diabetic retinopathy
grading models under domain shift by performing
counterfactual spatial node intervention directly on
intermediate feature representations.

The source model remains fully frozen. Each image is adapted
independently, without target labels, target-domain
statistics, gradients, parameter updates, or cross-image
adaptation state.

----------------------------------------------------------------------

Plug-and-Play Integration
-------------------------
CF-NODE is designed to be integrated into existing
PyTorch-based diabetic retinopathy grading models.

Your model should provide:

    • Classification logits
    • Spatial node features

A complete compatible implementation is provided in:

    model_example/model.py

----------------------------------------------------------------------

Main Functions
--------------

adapt_batch()
    Perform CF-NODE adaptation on a batch of images.

run_tta()
    Run CF-NODE over an entire test dataloader.

======================================================================
"""

import torch
import torch.nn.functional as F
import numpy as np

# =========================================================
# DEFAULT SOURCE PRIOR
#
# EyePACS empirical class distribution, computed from the
# full training set before balanced subsampling.
# =========================================================
DEFAULT_SOURCE_PRIOR = torch.tensor([
    0.734783,
    0.069550,
    0.150658,
    0.024853,
    0.020156,
]).float()

# =========================================================
# COUNTERFACTUAL INTERVENTION SCHEDULE
#
# Each pair is (node fraction, retained magnitude).
# =========================================================
DEFAULT_CF_LEVELS = (

    (0.02, 0.75),

    (0.05, 0.40),

    (0.10, 0.20),

    (0.15, 0.10),
)

# =========================================================
# AUGMENTATIONS
# =========================================================
AUGMENTATIONS = [

    lambda x: x,

    lambda x: torch.flip(x, dims=[3]),

    lambda x: torch.flip(x, dims=[2]),

    lambda x: torch.rot90(x, k=1, dims=[2, 3]),

    lambda x: torch.rot90(x, k=3, dims=[2, 3]),
]

# =========================================================
# CFG ACCESSOR
# =========================================================
def _cfg(cfg, key, default):

    return getattr(cfg, key, default)

# =========================================================
# RESOLVE SOURCE PRIOR
# =========================================================
def _resolve_source_prior(cfg, device):

    prior = _cfg(cfg, 'source_prior', None)

    if prior is None:

        prior = DEFAULT_SOURCE_PRIOR

    if not torch.is_tensor(prior):

        prior = torch.tensor(
            prior,
            dtype=torch.float32
        )

    prior = prior.float().to(device)

    return prior / (prior.sum() + 1e-8)

# =========================================================
# RESOLVE CF LEVELS
# =========================================================
def _resolve_cf_levels(cfg):

    levels = _cfg(cfg, 'cf_levels', None)

    if levels is None or len(levels) == 0:

        return DEFAULT_CF_LEVELS

    return levels

# =========================================================
# SAFE MODEL UNWRAP
# =========================================================
def get_base_model(model):

    return (
        model.module
        if isinstance(model, torch.nn.DataParallel)
        else model
    )

# =========================================================
# FORWARD WITH NODES
#
# Returns the class logits and the spatial node features.
# Any additional model outputs are ignored.
# =========================================================
def forward_with_nodes(model, x):

    out = model(x, return_nodes=True)

    if isinstance(out, tuple):

        if len(out) >= 2:

            logits = out[0]
            nodes = out[1]

        else:

            logits = out[0]
            nodes = None

    else:

        logits = out
        nodes = None

    return logits, nodes

# =========================================================
# EXPECTED GRADE
# =========================================================
def expected_grade_from_probs(probs):

    grades = torch.arange(
        probs.size(-1),
        device=probs.device
    ).float()

    return (probs * grades).sum(dim=-1)

# =========================================================
# ATTENTION SCORES
# =========================================================
def get_attention_scores(model, nodes):

    base = get_base_model(model)

    scores = base.attention_pool.attn(
        nodes
    ).squeeze(-1)

    return torch.softmax(scores, dim=1)

# =========================================================
# NODE-SET SIMILARITY
# =========================================================
def compute_cluster_consistency(nodes):

    nodes_norm = F.normalize(nodes, dim=-1)

    sim = torch.bmm(
        nodes_norm,
        nodes_norm.transpose(1, 2)
    ).mean(dim=-1)

    sim_min = sim.min(
        dim=1,
        keepdim=True
    )[0]

    sim_max = sim.max(
        dim=1,
        keepdim=True
    )[0]

    return (
        (sim - sim_min)
        / (sim_max - sim_min + 1e-8)
    )

# =========================================================
#  ESTIMATE SPATIAL NODE IMPORTANCE
#
# Purpose
# -------
# Computes the importance score for every spatial node by
# combining:
#
# • Attention weight
# • Feature magnitude
# • Node-set similarity
#
# Higher scores indicate stronger influence on the current
# prediction. This ranking selects the representations to be
# intervened on. It does not determine whether they contain
# pathological or domain-specific information.
# =========================================================
def compute_node_scores(attn_scores, nodes):

    strength = torch.norm(nodes, dim=-1)

    strength = strength / (
        strength.max(dim=1, keepdim=True)[0]
        + 1e-8
    )

    cluster = compute_cluster_consistency(nodes)

    return (
        0.50 * attn_scores
        + 0.25 * strength
        + 0.25 * cluster
    )

# =========================================================
# BUILD COUNTERFACTUAL SPATIAL NODES
#
# Purpose
# -------
# Creates multiple counterfactual feature representations
# by progressively attenuating the highest-ranked spatial
# nodes identified in the current image.
#
# Input
# -----
# nodes         : (B, N, D)
# node_scores   : (B, N)
#
# Output
# ------
# cf_nodes_all  : Counterfactual node representations
# num_cf        : Number of intervention levels
# =========================================================
def build_counterfactual_nodes(
    nodes,
    node_scores,
    cf_levels=DEFAULT_CF_LEVELS
):

    if cf_levels is None or len(cf_levels) == 0:

        cf_levels = DEFAULT_CF_LEVELS

    B, N, D = nodes.shape

    idx = torch.argsort(
        node_scores,
        dim=1,
        descending=True
    )

    cf_nodes_all = []

    for ratio, retained in cf_levels:

        k = max(int(N * ratio), 1)

        top_idx = idx[:, :k]

        cf_nodes = nodes.clone()

        scaling = torch.ones_like(cf_nodes)

        scaling.scatter_(
            1,
            top_idx.unsqueeze(-1).expand(-1, -1, D),
            retained
        )

        cf_nodes_all.append(
            cf_nodes * scaling
        )

    return torch.cat(cf_nodes_all, dim=0), len(cf_levels)

# =========================================================
# CLASSIFY CF NODES
# =========================================================
def classify_cf_nodes(model, cf_nodes, global_feat):

    base = get_base_model(model)

    # =====================================================
    # NODE POOL
    # =====================================================
    pooled_nodes = base.attention_pool(cf_nodes)

    # =====================================================
    # REBUILD FUSED FEATURE
    # =====================================================
    fused = torch.cat(
        [
            pooled_nodes,
            global_feat
        ],
        dim=1
    )

    # =====================================================
    # FUSION BLOCK
    # =====================================================
    fused = base.fusion_block(fused)

    # =====================================================
    # DROPOUT
    # =====================================================
    fused = base.feature_dropout(fused)

    # =====================================================
    # CLASSIFIER
    # =====================================================
    logits = base.classifier(fused)

    return logits

# =========================================================
# STABLE INTERVENTION SENSITIVITY
#
# Mean one-sided expected-grade reduction, normalized by its
# variation across intervention levels.
# =========================================================
def compute_stable_intervention_sensitivity(exp_orig, exp_cf):

    drops = F.relu(
        exp_orig.unsqueeze(0) - exp_cf
    )

    mean_drop = drops.mean(dim=0)

    std_drop = drops.std(dim=0)

    return mean_drop / (std_drop + 0.15)

# =========================================================
# SOURCE PRIOR CORRECTION  (v1 formula — DO NOT CHANGE)
# =========================================================
def apply_source_prior_correction(probs, cfg):

    prior = _resolve_source_prior(
        cfg,
        probs.device
    )

    strength = _cfg(
        cfg,
        'prior_correction_strength',
        0.18
    )

    logits = torch.log(probs + 1e-8)

    logits = logits + (
        strength * torch.log(prior + 1e-8)
    )

    return F.softmax(logits, dim=-1)

# =========================================================
# SAFE NORMALIZE
# =========================================================
def safe_normalize(probs):

    probs = torch.nan_to_num(probs)

    probs = torch.clamp(
        probs,
        min=1e-6
    )

    probs = probs / (
        probs.sum(dim=-1, keepdim=True)
        + 1e-8
    )

    return probs

# =========================================================
# SINGLE VIEW
# =========================================================
def _cf_node_single_view(model, images, cfg):

    base_temp = _cfg(cfg, 'temp', 1.05)

    B = images.size(0)

    base = get_base_model(model)

    # =====================================================
    # ORIGINAL FORWARD
    # =====================================================
    logits_orig, nodes = forward_with_nodes(
        model,
        images
    )

    probs_orig = F.softmax(
        logits_orig / base_temp,
        dim=-1
    )

    exp_orig = expected_grade_from_probs(
        probs_orig
    )

    # =====================================================
    # FALLBACK
    # =====================================================
    if nodes is None:

        return probs_orig

    # =====================================================
    # RECOMPUTE GLOBAL FEATURES
    # =====================================================
    feat = base.features(images)

    global_feat = base.gem_pool(feat)

    global_feat = base.global_proj(global_feat)

    # =====================================================
    # NODE IMPORTANCE
    # =====================================================
    attn_scores = get_attention_scores(
        model,
        nodes
    )

    node_scores = compute_node_scores(
        attn_scores,
        nodes
    )

    # =====================================================
    # BUILD CF NODES
    # =====================================================
    cf_levels = _resolve_cf_levels(cfg)

    cf_nodes_all, num_cf = build_counterfactual_nodes(
        nodes,
        node_scores,
        cf_levels=cf_levels
    )

    # =====================================================
    # REPEAT GLOBAL FEATURES
    # =====================================================
    global_feat_repeat = global_feat.repeat(
        num_cf,
        1
    )

    # =====================================================
    # CF CLASSIFICATION
    # =====================================================
    logits_cf_all = classify_cf_nodes(
        model,
        cf_nodes_all,
        global_feat_repeat
    )

    probs_cf_all = F.softmax(
        logits_cf_all / base_temp,
        dim=-1
    )

    exp_cf = expected_grade_from_probs(
        probs_cf_all
    ).view(num_cf, B)

    # =====================================================
    # STABLE SENSITIVITY SCORE
    # =====================================================
    stable_drop = compute_stable_intervention_sensitivity(
        exp_orig,
        exp_cf
    )

    # =====================================================
    # SENSITIVITY-DEPENDENT TEMPERATURE
    # =====================================================
    sample_temp = (
        base_temp
        - 0.22 * stable_drop
    )

    sample_temp = torch.clamp(
        sample_temp,
        min=0.72,
        max=1.08
    )

    # =====================================================
    # BOUNDED CALIBRATION
    # =====================================================
    final_logits = (
        torch.log(probs_orig + 1e-8)
        / sample_temp.unsqueeze(-1)
    )

    return F.softmax(final_logits, dim=-1)

# =========================================================
# ADAPT BATCH  (argmax — v1 behaviour)
# =========================================================
@torch.no_grad()
def adapt_batch(model, images, cfg, return_probs=False):

    model.eval()

    view_probs_list = []

    for aug_fn in AUGMENTATIONS:

        view_probs = _cf_node_single_view(
            model,
            aug_fn(images),
            cfg
        )

        view_probs_list.append(view_probs)

    avg_probs = torch.stack(
        view_probs_list,
        dim=0
    ).mean(dim=0)

    # =====================================================
    # SOURCE PRIOR CORRECTION
    # =====================================================
    final_probs = apply_source_prior_correction(
        avg_probs,
        cfg
    )

    final_probs = safe_normalize(final_probs)

    # =====================================================
    # ARGMAX (v1)
    # =====================================================
    final_preds = torch.argmax(
        final_probs,
        dim=-1
    )

    # =====================================================
    # RANKING SCORE  (posterior expected grade)
    # =====================================================
    ranking_scores = expected_grade_from_probs(
        final_probs
    )

    preds_np = final_preds.cpu().numpy().astype(np.int64)

    probs_np = final_probs.cpu().numpy().astype(np.float32)

    ranking_np = ranking_scores.cpu().numpy().astype(np.float32)

    if return_probs:

        return (
            preds_np,
            probs_np,
            ranking_np
        )

    return preds_np

# =========================================================
# RUN TTA
# =========================================================
def run_tta(model, loader, cfg, return_probs=False):

    preds_all = []

    probs_all = []

    ranking_all = []

    labels_all = []

    device = next(model.parameters()).device

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        if return_probs:

            preds, probs, ranking = adapt_batch(
                model,
                images,
                cfg,
                return_probs=True
            )

            probs_all.append(probs)

            ranking_all.append(ranking)

        else:

            preds = adapt_batch(
                model,
                images,
                cfg
            )

        preds_all.extend(preds)

        labels_all.extend(labels.numpy())

    preds_all = np.array(
        preds_all,
        dtype=np.int64
    )

    labels_all = np.array(
        labels_all,
        dtype=np.int64
    )

    if return_probs:

        probs_all = np.concatenate(
            probs_all,
            axis=0
        ).astype(np.float32)

        ranking_all = np.concatenate(
            ranking_all,
            axis=0
        ).astype(np.float32)

        return (
            preds_all,
            labels_all,
            probs_all,
            ranking_all
        )

    return preds_all, labels_all
