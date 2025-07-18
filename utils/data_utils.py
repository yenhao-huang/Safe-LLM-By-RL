from datasets import load_dataset

def download_anthropic_hh(save_dir="./"):
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
