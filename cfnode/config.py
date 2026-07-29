"""
===============================================================
CF-NODE Configuration
This file contains all configurable hyperparameters used by the
CF-NODE algorithm.
These parameters control the behaviour of counterfactual node
intervention, intervention-sensitivity estimation,
sensitivity-dependent calibration, and source prior correction.
Users typically only need to modify this file to experiment
with different CF-NODE settings.
For implementation details, see:
    cfnode/cfnode.py
===============================================================
"""
class CFNodeConfig:
    """
    Configuration for the CF-NODE algorithm.
    """
    # ==========================================================
    # TARGET DATASET
    # ==========================================================
    #
    # Used only for logging and bookkeeping. CF-NODE applies the
    # same settings to every target domain.
    #
    # This value can be overwritten dynamically in your own
    # evaluation script.
    #
    target_dataset = "messidor"
    target_name = "messidor"
    # ==========================================================
    # ENABLE / DISABLE COMPONENTS
    # ==========================================================
    use_cf_adaptation = True
    use_prior_correction = True
    # ==========================================================
    # SOURCE DOMAIN PRIOR
    # ==========================================================
    #
    # EyePACS empirical class distribution, computed from the
    # full training set before balanced subsampling.
    #
    source_prior = [
        0.734783,
        0.069550,
        0.150658,
        0.024853,
        0.020156,
    ]
    # ==========================================================
    # SOURCE PRIOR CORRECTION
    # ==========================================================
    prior_correction_strength = 0.18
    # ==========================================================
    # BASE SOFTMAX TEMPERATURE
    # ==========================================================
    temp = 1.05
    # ==========================================================
    # COUNTERFACTUAL INTERVENTION LEVELS
    # ==========================================================
    #
    # Each pair is (node fraction, retained magnitude).
    #
    # None uses the default intervention schedule implemented
    # inside CF-NODE.
    #
    cf_levels = None
