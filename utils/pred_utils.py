import json
import os

def save_predictions_to_jsonl(
    predictions,
    test_dataset,
    tokenizer,
    output_dir="predicts"
):
    input_ids = test_dataset["input_ids"]
    decoded_inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "predictions.jsonl")

    with open(jsonl_path, mode="w", encoding="utf-8") as f:
        for input_text, pred_text in zip(decoded_inputs, predictions):
            record = {
                "input_text": input_text,
                "predicted_text": pred_text
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved predictions to: {jsonl_path}")
