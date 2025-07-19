from datasets import load_dataset
from transformers import AutoTokenizer

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


def select_partial_data(raw_data, train_size=1000, test_size=100, seed=42):
    """Select partial samples from train and test splits."""
    raw_data["train"] = raw_data["train"].shuffle(seed=seed).select(range(train_size))
    raw_data["test"] = raw_data["test"].shuffle(seed=seed).select(range(test_size))

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