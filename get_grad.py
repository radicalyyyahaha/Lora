import os
import torch
import numpy as np
from sklearn.random_projection import SparseRandomProjection
from sklearn.cluster import KMeans
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
import contextlib
from functools import partial
from typing import List, Union

from torch.utils.data import DataLoader
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = "./gpt2" 
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

adapter_model_path = "./ckpts/checkpoint-49784" 
model = PeftModel.from_pretrained(model, adapter_model_path)
model.to(device) 
print("Model load to device")


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


def compute_lora_gradients(data_point):
    """Compute gradients for a single data point"""
    model.train()

    for param in model.parameters():
        param.requires_grad = True

    model.zero_grad()

    inputs = {
    'input_ids': data_point["input_ids"].unsqueeze(0).to(device).clone().detach(),
    'attention_mask': data_point["attention_mask"].unsqueeze(0).to(device).clone().detach(),
    'labels': data_point["labels"].unsqueeze(0).to(device).clone().detach()
    }

    outputs = model(**inputs)
    loss = outputs.loss

    loss.backward()

    lora_gradients = {}
    for name, param in model.named_parameters():
        if "lora" in name and param.grad is not None:
            lora_gradients[name] = param.grad.detach().cpu().numpy()

    return lora_gradients

data_files = [
    "data/cot_data.jsonl", 
    "data/flan_v2_data.jsonl", 
    "data/dolly_data.jsonl", 
    "data/oasst1_data.jsonl"
]
train_dataset = get_training_dataset(data_files, tokenizer, max_seq_length=1024, sample_percentage=0.05, seed=42)

gradient_repository = "./grad_repository"
os.makedirs(gradient_repository, exist_ok=True)

rp = SparseRandomProjection(n_components=128)

for idx, data_point in enumerate(tqdm(train_dataset, desc="Computing LoRA Gradients for Each Data Point")):

    gradients = compute_lora_gradients(data_point)
    flat_gradients = np.concatenate([v.flatten() for v in gradients.values()])
    low_dim_gradients = rp.fit_transform(flat_gradients.reshape(1, -1))

    np.save(os.path.join(gradient_repository, f"low_dim_grad_point_{idx}.npy"), low_dim_gradients)

gradient_files = [os.path.join(gradient_repository, f"low_dim_grad_point_{i}.npy") for i in range(len(train_dataset))]
all_low_dim_gradients = [np.load(file) for file in gradient_files]
all_low_dim_gradients = np.vstack(all_low_dim_gradients)

np.save(os.path.join(gradient_repository, "final_low_dim_grad_matrix.npy"), all_low_dim_gradients)

print("LoRA gradients computed and saved successfully.")