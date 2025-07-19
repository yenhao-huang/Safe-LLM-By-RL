from transformers import AutoModel


def build_compute_metrics(tokenizer, detox_model):

    def compute_metrics(decoded_texts):
        print(decoded_texts)
        # Detoxify toxicity scores
        toxicity_scores = [detox_model.predict(text)["toxicity"] for text in decoded_texts]
        avg_toxicity = sum(toxicity_scores) / len(toxicity_scores)

        return {
            "avg_toxicity": avg_toxicity
        }

    return compute_metrics
