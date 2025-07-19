import os
import sys

# 專案根目錄 = scripts 的上一層
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import argparse
from detoxify import Detoxify
from utils import utils, data_utils, model_utils, metric_utils, pred_utils


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate")
    parser.add_argument("--text_col", type=str, default="chosen")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--n_labels", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--enable_lora", type=bool, default=False)
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--eval_batch", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--log_dir", type=str, default="./logs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir, log_dir = utils.make_dir_with_timestamp(args.output_dir, args.log_dir)
    raw_data = data_utils.download_anthropic_hh()

    data_utils.select_partial_data(
        raw_data, 
        train_size=10, 
        test_size=8000, 
        seed=42
    )

    tonkenized_data, tokenizer = data_utils.data_preprocess(
        raw_data, 
        args.model, 
        args.text_col, 
        args.label_col, 
        args.seq_len
    )

    _, _, test_dataset = data_utils.split_dataset(tonkenized_data)
    eval_agent = model_utils.set_eval_agent(
        args.model, 
        args.enable_lora,
    )
    
    predictions = model_utils.inference(
        eval_agent, 
        tokenizer,
        test_dataset, 
        args.text_col,
        args.eval_batch,
        max_new_tokens=50,
    )

    detox_model = Detoxify('original')
    compute_metrics = metric_utils.build_compute_metrics(tokenizer, detox_model)
    results = compute_metrics(predictions)
    print(results)
    pred_utils.save_predictions_to_jsonl(predictions, test_dataset, tokenizer)