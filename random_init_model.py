import json
from transformers import AutoModelForCausalLM, AutoConfig, MixtralConfig, AutoTokenizer, LlamaConfig

moe_config_str = """{
  "architectures": [
    "MixtralForCausalLM"
  ],
  "attention_dropout": 0.0,
  "attention_bias": false,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "max_position_embeddings": 2048,
  "model_type": "mixtral",
  "num_attention_heads": 16,
  "num_hidden_layers": 16,
  "num_key_value_heads": 16,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.35.0",
  "use_cache": true,
  "vocab_size": 32000,
  "num_experts_per_tok": 2,
  "num_local_experts": 8,
  "output_router_logits": false,
  "router_aux_loss_coef": 0.02
}"""

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo-1b")

cosmo_config = AutoConfig.from_pretrained("HuggingFaceTB/cosmo-1b")

dense_model = AutoModelForCausalLM.from_config(cosmo_config)
n_params = sum({p.data_ptr(): p.numel() for p in dense_model.parameters()}.values())

print(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

moe_config = json.loads(moe_config_str)
moe_config = MixtralConfig.from_dict(moe_config)
moe_model = AutoModelForCausalLM.from_config(moe_config)
n_params = sum({p.data_ptr(): p.numel() for p in moe_model.parameters()}.values())

print(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

small_dense_config = json.loads(moe_config_str)
small_dense_config.pop("num_experts_per_tok")
small_dense_config.pop("num_local_experts")
small_dense_config.pop("output_router_logits")
small_dense_config.pop("router_aux_loss_coef")
small_dense_config['architectures'] = ["LlamaForCausalLM"]
small_dense_config = LlamaConfig.from_dict(small_dense_config)
small_dense_model = AutoModelForCausalLM.from_config(small_dense_config)
n_params = sum({p.data_ptr(): p.numel() for p in small_dense_model.parameters()}.values())

print(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

# dense_model.push_to_hub("yentinglin/cosmo-1b-random-init")
# moe_model.push_to_hub("yentinglin/cosmo-8x220M-random-init")
# tokenizer.push_to_hub("yentinglin/cosmo-1b-random-init")
# tokenizer.push_to_hub("yentinglin/cosmo-8x220M-random-init")
# small_dense_model.push_to_hub("yentinglin/cosmo-220M-random-init")
# tokenizer.push_to_hub("yentinglin/cosmo-220M-random-init")
