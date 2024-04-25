""" Example python script to generate a YAML config file which can be used to run a training with nanotron. Refer to "examples" section in the `/README.md` for more information."""
import os
from dataclasses import dataclass
from typing import Optional

from nanotron.config import (
    AdamWOptimizerArgs,
    CheckpointsArgs,
    Config,
    DataArgs,
    DatasetStageArgs,
    GeneralArgs,
    LoggingArgs,
    LRSchedulerArgs,
    ModelArgs,
    OptimizerArgs,
    ParallelismArgs,
    RandomInit,
    TokenizerArgs,
    TokensArgs,
    LlamaConfig
)
from nanotron.config.config import PretrainDatasetsArgs
from nanotron.logging import human_format


@dataclass
class LlaMoEConfig(LlamaConfig):
    ## MoE specific
    # Number of experts per Sparse MLP layer.
    moe_num_experts: int = 1
    # the number of experts to root per-token, can be also interpreted as the `top-p` routing parameter
    num_experts_per_tok: int = 1
    moe_capacity_factor: int = 1


model_config = LlaMoEConfig(
    # Config for a 52M llama model
    num_hidden_layers=16,
    hidden_size=1024,
    num_attention_heads=16,
    num_key_value_heads=16,
    rms_norm_eps=1e-05,
    intermediate_size=1024 * 4,
    max_position_embeddings=2048,
    tie_word_embeddings=False,
    vocab_size=32000,
    moe_num_experts=8,
    num_experts_per_tok=2,
)

num_params = human_format(
    model_config.vocab_size * model_config.hidden_size * 2
    + model_config.num_hidden_layers
    * (
        # router
        model_config.moe_num_experts * model_config.hidden_size
        +
        # expert
        model_config.moe_num_experts * 3 * model_config.hidden_size * model_config.intermediate_size
        +
        # layernorm
        2 * model_config.hidden_size
        +
        4 * model_config.hidden_size * model_config.hidden_size
    )
).replace(".", "p")
num_active_params = human_format(
    model_config.vocab_size * model_config.hidden_size * 2
    + model_config.num_hidden_layers
    * (
        # router
        model_config.moe_num_experts * model_config.hidden_size
        +
        # expert
        model_config.num_experts_per_tok * 3 * model_config.hidden_size * model_config.intermediate_size
        +
        # layernorm
        2 * model_config.hidden_size
        +
        4 * model_config.hidden_size * model_config.hidden_size
    )
).replace(".", "p")

print(f"Model has {num_params} total parameters and {num_active_params} active parameters")

SEED = 42

learning_rate = LRSchedulerArgs(
    learning_rate=3e-4, lr_warmup_steps=100, lr_warmup_style="linear", lr_decay_style="cosine", min_decay_lr=1e-5
)

optimizer = OptimizerArgs(
    zero_stage=0,
    weight_decay=0.01,
    clip_grad=1.0,
    accumulate_grad_in_fp32=False,
    learning_rate_scheduler=learning_rate,
    optimizer_factory=AdamWOptimizerArgs(
        adam_eps=1e-08,
        adam_beta1=0.9,
        adam_beta2=0.95,
        torch_adam_is_fused=True,
    ),
)

parallelism = ParallelismArgs(
    dp=32,
    pp=1,
    tp=1,
    expert_parallel_size=1,
    pp_engine="1f1b",
    tp_mode="ALL_REDUCE",
    tp_linear_async_communication=False,
)

assert (
    model_config.moe_num_experts % parallelism.expert_parallel_size == 0
), "Number of experts must be divisible by expert_parallel_size"

tokens = TokensArgs(sequence_length=2048, train_steps=1918, micro_batch_size=16, batch_accumulation_per_replica=8)

data = DataArgs(
    seed=SEED,
    num_loading_workers=1,
    # dataset=None
    dataset=PretrainDatasetsArgs(
        hf_dataset_or_datasets="HuggingFaceTB/cosmopedia_6M",
        # hf_dataset_config_name="auto_math_text",
        hf_dataset_splits="train",
        text_column_name="text",
        dataset_processing_num_proc_per_process=128,
    ),
)


checkpoints_path = os.path.dirname(os.path.dirname(__file__)) + "/checkpoints"
os.makedirs(checkpoints_path, exist_ok=True)

config = Config(
    general=GeneralArgs(project="moe", run="llamoe-cosmopedia", seed=SEED),
    checkpoints=CheckpointsArgs(
        checkpoints_path=checkpoints_path,
        checkpoint_interval=100000,
        save_initial_state=True,
        # resume_checkpoint_path=checkpoints_path,
    ),
    parallelism=parallelism,
    model=ModelArgs(init_method=RandomInit(std=0.025), model_config=model_config),
    tokenizer=TokenizerArgs("meta-llama/Llama-2-7b-hf"),
    optimizer=optimizer,
    logging=LoggingArgs(),
    tokens=tokens,
    data_stages=[
        DatasetStageArgs(name="Stable Training Stage", start_training_step=1, data=data),
        DatasetStageArgs(name="Annealing Phase", start_training_step=10, data=data),
    ],
)

if __name__ == "__main__":
    dir = os.path.dirname(__file__)

    # Save config as YAML file
    filename = os.path.basename(__file__).replace(".py", ".yaml")
    config.save_as_yaml(f"{dir}/{filename}")
    print(f"Config saved as {dir}/{filename}")
    # You can now train a model with this config using `/run_train.py`
