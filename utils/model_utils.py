from transformers import  AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
import torch
from tqdm import tqdm

from peft import PeftConfig, PeftModel, get_peft_model, LoraConfig, TaskType
from utils.utils import CUDAMemoryTrackerCallback
from torch.utils.data import DataLoader

def set_eval_agent(
    model_name, 
    enable_lora=False
):

    if enable_lora:
        peft_config = PeftConfig.from_pretrained(model_name)
        base_model =  AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)
        model = PeftModel.from_pretrained(base_model, model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name
        )

    return model

def inference(
    model, 
    tokenizer, 
    dataset, 
    text_col,
    batch_size=1,
    max_new_tokens=50
):

    """
    Simple inference loop for text generation without Trainer.
    
    Args:
        model: HuggingFace model (should be in eval mode with correct LM head)
        tokenizer: Corresponding tokenizer
        dataset: Iterable of dict with "text" key or HuggingFace Dataset
        batch_size: Batch size for inference
        max_new_tokens: Number of tokens to generate
    
    Returns:
        List of decoded generated text strings
    """

    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size)

    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating"):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            # Decode predictions into text
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results.extend(decoded)

    return results

