"""
UnchartedLLM: Controllable-Length Text Generation Framework
xVal (Continuous Value Encoding) with Two-Stage Training
Based on Qwen3-8B with Remaining Token Countdown Mechanism
Version: 3.1.0 - Step Noise
"""

from .core import (
    value_to_token_id,
    token_id_to_value,
    extend_tokenizer_safe,
    initialize_num_embedding,
    value_to_xval_embedding,
    batch_value_to_xval_embedding,
    sample_noisy_target,
    sample_relative_noisy_target,
    print_xval_info,
    MAX_TARGET_VALUE,
    XVAL_ALPHA,
    NUM_SPECIAL_TOKENS,
)

from .model import (
    UnchartedModelWrapper,
)

from .loss import (
    compute_loss_stage1,
    compute_loss_stage2,
    compute_text_loss,
    compute_regression_loss,
    compute_shift_loss,
)

from .inference import (
    RemainingState,
    inference_step,
    generate_with_countdown,
    batch_generate,
    evaluate_length_control,
)

from .dataset import (
    FrictionHybridDataset,
    collate_friction_batch
)

__version__ = "3.1.0"

__all__ = [
    # Core
    "value_to_token_id",
    "token_id_to_value",
    "extend_tokenizer_safe",
    "initialize_num_embedding",
    "value_to_xval_embedding",
    "batch_value_to_xval_embedding",
    "sample_noisy_target",
    "sample_relative_noisy_target",
    "print_xval_info",
    "MAX_TARGET_VALUE",
    "XVAL_ALPHA",
    "NUM_SPECIAL_TOKENS",

    # Model
    "UnchartedModelWrapper",

    # Loss
    "compute_loss_stage1",
    "compute_loss_stage2",
    "compute_text_loss",
    "compute_regression_loss",
    "compute_shift_loss",

    # Inference
    "RemainingState",
    "inference_step",
    "generate_with_countdown",
    "batch_generate",
    "evaluate_length_control",

    # Dataset
    "FrictionHybridDataset",
    "collate_friction_batch",
]
