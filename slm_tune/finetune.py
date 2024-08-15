
# %%capture

# # Installs Unsloth, Xformers (Flash Attention) and all other packages!
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes

# * Ref: https://www.youtube.com/watch?v=UZbp5TsNJTw

# TODO: Build the dataset with the input and output fields; Instruction can be fixed in the prompt

'''
This file use to finetune Qwen2 model. It uses the Alpaca API to generate summaries.
'''
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048
dtype = None
load_in_4bit = True

fourbit_models = [
    "unsloth/Qwen2-0.5b-bnb-4bit",
]

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2-0.5B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)


model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples, size=100):
    instruction = f"Summarize below {size} bullets paragraph into {size} single-sentence summaries as 1-on-1 mapping."
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []

    for _input, output in zip(inputs, outputs):
        text = alpaca_prompt.format(instruction, _input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}
pass

from datasets import load_dataset, Dataset
import pandas as pd
# Load the local CSV file
try:
    # Read the CSV file in chunks
    chunks = pd.read_csv("train.csv", engine='python', chunksize=10000)
    df = pd.concat(chunks)
    print("CSV file loaded successfully.")
except pd.errors.ParserError as e:
    print(f"Error reading CSV file: {e}")
    exit(1)

# Convert the DataFrame to a Dataset
dataset = Dataset.from_pandas(df)

# dataset = load_dataset('csv', data_files='./train.csv')

dataset = dataset.map(formatting_prompts_func, batched = True)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,

        # Use num_train_epochs = 1, warmup_ratio for full training runs!
        warmup_steps = 20,
        max_steps = 120,

        learning_rate = 5e-5,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer_stats = trainer.train()
