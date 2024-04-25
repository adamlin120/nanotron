from datasets import load_dataset, concatenate_datasets

ds = [
    # load_dataset("yentinglin/ultra_200k_llamaformat", split="train_sft"),
    # load_dataset("bigcode/starcoderdata", data_dir="jupyter-scripts-dedup-filtered", split="train"),
    # load_dataset("bigcode/starcoderdata", data_dir="jupyter-structured-clean-dedup", split="train"),
    # load_dataset("math-ai/AutoMathText", "code-python-0.60-to-1.00", split='train', num_proc=12),
    # load_dataset("math-ai/AutoMathText", "web-0.50-to-1.00", split='train', num_proc=12),
    load_dataset("HuggingFaceTB/cosmopedia", "stories", split="train", num_proc=12),
    load_dataset("HuggingFaceTB/cosmopedia", "stories", split="train", num_proc=12),
    load_dataset("HuggingFaceTB/cosmopedia", "khanacademy", split="train", num_proc=12),
    load_dataset("HuggingFaceTB/cosmopedia", "khanacademy", split="train", num_proc=12),
    load_dataset("HuggingFaceTB/cosmopedia", "auto_math_text", split="train", num_proc=12),
    load_dataset("HuggingFaceTB/cosmopedia", "auto_math_text", split="train", num_proc=12),
    load_dataset("HuggingFaceTB/cosmopedia", "openstax", split="train", num_proc=12),
    load_dataset("HuggingFaceTB/cosmopedia", "stanford", split="train", num_proc=12),
    load_dataset("HuggingFaceTB/cosmopedia", "web_samples_v1", split="train", num_proc=12),
    load_dataset("HuggingFaceTB/cosmopedia", "web_samples_v2", split="train", num_proc=12),
    load_dataset("HuggingFaceTB/cosmopedia", "wikihow", split="train", num_proc=12),
]
for d in ds:
    print(d)

ds = concatenate_datasets(ds)

import ipdb
ipdb.set_trace()
