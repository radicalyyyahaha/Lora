import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import contextlib
from functools import partial
from typing import List, Union

torch.manual_seed(1337)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# initialize
model_name = "./gpt2"  
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# pad_token
tokenizer.pad_token = tokenizer.eos_token  

# LoRA args
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["attn.c_attn"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def get_training_dataset(train_files: List[str], tokenizer, max_seq_length, sample_percentage=1.0, seed=0):
    """ Get training dataset with a specified seed """
    raw_datasets = load_raw_dataset(train_files, sample_percentage=sample_percentage, seed=seed)
    lm_datasets = encode_data(raw_datasets, tokenizer, max_seq_length)
    return lm_datasets

def load_raw_dataset(train_files: Union[List[str], str], sample_size=None, sample_percentage=1.0, seed=0):
    """ Load raw dataset """
    if isinstance(train_files, str):
        train_files = [train_files]
    processed_datasets = load_dataset("json", data_files=train_files)["train"]
    if sample_size is None:
        sample_size = int(len(processed_datasets) * sample_percentage)

    if sample_size == len(processed_datasets):
        return processed_datasets  # Not shuffle

    with temp_seed(seed):
        index = np.random.permutation(len(processed_datasets))[:sample_size]

    sampled_dataset = processed_datasets.select(index)
    return sampled_dataset

def encode_data(raw_datasets, tokenizer, max_seq_length, processing_num_workers=10, overwrite_cache=False, func_name="encode_with_messages_format"):
    """ Encode data with the specified tokenizer and chat format. """
    if "input_ids" in raw_datasets.features:
        return raw_datasets
    encode_function = get_encode_function(raw_datasets, tokenizer, max_seq_length, func_name)
    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=processing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc="Tokenizing and reformatting instruction data",
    )
    lm_datasets.set_format(type="pt")
    return lm_datasets

def get_encode_function(raw_datasets, tokenizer, max_seq_length, func="encode_with_messages_format"):
    """ Get encode function based on the dataset. """
    if "prompt" in raw_datasets.column_names and "completion" in raw_datasets.column_names:
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
    elif "messages" in raw_datasets.column_names:
        encode_function = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
    else:
        raise ValueError("You need to have either 'prompt'&'completion' or 'messages' in your column names.")
    return encode_function


def encode_with_prompt_completion_format(example, tokenizer, max_seq_length):
    """处理带有 'prompt' 和 'completion' 的数据格式"""
    if not example['prompt'].endswith((' ', '\n', '\t')) and not example['completion'].startswith((' ', '\n', '\t')):
        example_text = example['prompt'] + ' ' + example['completion']
    else:
        example_text = example['prompt'] + example['completion']
    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(
        example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(
        example['prompt'], return_tensors='pt', max_length=max_seq_length, truncation=True)
    labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }

def encode_with_messages_format(example, tokenizer, max_seq_length):
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')

    example_text = concat_messages(messages, tokenizer)
    tokenized_example = tokenizer(
        example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    
    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            message_start_idx = tokenizer(
                concat_messages(messages[:message_idx], tokenizer), return_tensors='pt', max_length=max_seq_length, truncation=True
            ).input_ids.shape[1]
            message_end_idx = tokenizer(
                concat_messages(messages[:message_idx + 1], tokenizer),
                return_tensors='pt',
                max_length=max_seq_length,
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }

def concat_messages(messages, tokenizer):
    message_text = ""
    for message in messages:
        if message["role"] == "system":
            message_text += "<|system|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "user":
            message_text += "<|user|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "assistant":
            message_text += "<|assistant|>\n" + \
                message["content"].strip() + tokenizer.eos_token + "\n"
        else:
            raise ValueError("Invalid role: {}".format(message["role"]))
    return message_text


data_files = [
    "data/cot_data.jsonl", 
    "data/flan_v2_data.jsonl", 
    "data/dolly_data.jsonl", 
    "data/oasst1_data.jsonl"
]
train_dataset = get_training_dataset(data_files, tokenizer, max_seq_length=1024, sample_percentage=0.05, seed=42)

# save ckpt
output_dir = "./ckpts"
os.makedirs(output_dir, exist_ok=True)

# train args
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="no",
    learning_rate=2e-05,
    per_device_train_batch_size=1,
    num_train_epochs=4,
    weight_decay=0.01,
    save_total_limit=3,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    bf16=True, 
)


trainer = Trainer(
    model=model.to(device),
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

trainer.train()