import json
import os

from datasets import DatasetDict, Dataset, load_dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split


def download_anthropic_hh(save_dir="./data"):
    """
    Downloads the Anthropic HH-RLHF dataset and saves train/test splits as JSONL files.

    Args:
        save_dir (str): Directory to save the dataset files.

    Returns:
        dataset (DatasetDict): The loaded dataset splits.
    """
    print("Downloading Anthropic HH-RLHF dataset...")
    dataset = load_dataset("Anthropic/hh-rlhf")

    print("Saving datasets to:", save_dir)
    dataset['train'].to_json(f"{save_dir}/hh_rlhf_train.jsonl")
    dataset['test'].to_json(f"{save_dir}/hh_rlhf_test.jsonl")

    print("Download and save complete.")
    return dataset


def load_toxicity_subset(file_path="data/toxic_subset_from_hh.json"):
    """
    Loads the toxicity subset dataset from a JSON file and returns a DatasetDict with 'train' and 'test' splits.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dataset_dict (DatasetDict): Huggingface DatasetDict with 'train' and 'test' splits.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset = Dataset.from_list(data)

    return DatasetDict({
        "train": dataset,
        "test": dataset
    })

def load_real_toxic_prompt(
    train_path="data/part_real_toxic_prompt_train.jsonl",
    test_path="data/part_real_toxic_prompt_test.jsonl"
    ):

    def load_jsonl(file_path):
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    train_data = load_jsonl(train_path)
    test_data = load_jsonl(test_path)

    return DatasetDict({
        "train": Dataset.from_list(train_data),
        "test": Dataset.from_list(test_data)
    })

class Real_Toxicity_Prompt_Dataset():
    def __init__(self):
        pass

    def download_real_toxic_prompt(self, save_dir="./data"):
        dataset = load_dataset("allenai/real-toxicity-prompts", split="train")
        
        print("Saving datasets to:", save_dir)
        dataset.to_json(f"{save_dir}/real_toxicity_prompts.jsonl")
        
        print("Download and save complete.")
        return DatasetDict({
            "train": dataset,
            "test": dataset
        })

    def get_toxic_prompts(self, threshold=0.5, max_samples=100):
        dataset = load_dataset("allenai/real-toxicity-prompts", split="train")
        
        toxic_samples = []
        for example in dataset:
            score = example.get('continuation', {}).get('toxicity', 0.0)
            if isinstance(score, (int, float)):
                if score > threshold:
                    toxic_samples.append({
                        "prompt": example["prompt"],
                        "continuation": example["continuation"],
                        "text": example["prompt"]["text"],
                        "toxicity_score": score
                    })
                    '''
                    print(example["prompt"]["text"])
                    print("\n")
                    print(example["continuation"]["text"])
                    print("\n")
                    print(score)
                    print("=" * 80)
                    '''
                    if len(toxic_samples) >= max_samples:
                        break
                    
        return toxic_samples

    def save_toxic_samples_split(
        self, 
        samples, 
        save_dir="data", 
        filename_prefix="part_real_toxic_prompt", 
        test_size=0.5, 
        random_state=42
        ):

        train_samples, test_samples = train_test_split(samples, test_size=test_size, random_state=random_state)

        for split_name, split_samples in [("train", train_samples), ("test", test_samples)]:
            save_path = os.path.join(save_dir, f"{filename_prefix}_{split_name}.jsonl")
            with open(save_path, "w", encoding="utf-8") as f:
                for sample in split_samples:
                    json.dump(sample, f, ensure_ascii=False)
                    f.write("\n")
            print(f"Saved {len(split_samples)} {split_name} samples to {save_path}")

def select_partial_data(raw_data, train_size=1000, test_size=100, seed=42):
    """Select partial samples from train and test splits."""
    if len(raw_data["train"]) > train_size:
        raw_data["train"] = raw_data["train"].shuffle(seed=seed).select(range(train_size))
    
    if len(raw_data["test"]) > test_size:
        raw_data["test"] = raw_data["test"].shuffle(seed=seed).select(range(test_size))

from datasets import Dataset, DatasetDict

def prompt_engineering(raw_data):
    """
    Process a Huggingface DatasetDict and rewrite input_text with a prompt template.

    Args:
        raw_data (DatasetDict): {"train": Dataset, "test": Dataset}

    Returns:
        DatasetDict: {"train": Dataset, "test": Dataset} with "prompt" column
    """
    def process_example(example):
        input_text = example.get("text", "").strip()
        #prompt = f"Continue writing the following paragraph naturally:\n\n{input_text}"
        #prompt = (
        #    f"Please write a complete and coherent paragraph based on the following beginning:\n\n"
        #    f"{input_text}\n\n"
        #    f"Make sure to continue naturally and conclude the paragraph."
        #)
        example["prompt"] = prompt
        return example

    processed_data = DatasetDict({
        split_name: dataset.map(process_example)
        for split_name, dataset in raw_data.items()
    })

    return processed_data


def data_preprocess(
    raw_data, 
    model_name, 
    text_col, 
    label_col, 
    sequence_len=16384
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        texts = [x if isinstance(x, str) else "" for x in batch[text_col]]
        return tokenizer(
            texts, 
            padding="max_length", 
            truncation=True,
            max_length=sequence_len,
            )

    tokenized_data = raw_data.map(tokenize, batched=True)
    tokenized_data.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return tokenized_data, tokenizer

def split_dataset(tonkenized_data):
    return tonkenized_data["train"], tonkenized_data["test"], tonkenized_data["test"]