import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from utils.data_utils import download_anthropic_hh, Real_Toxicity_Prompt_Dataset

if __name__ == "__main__":
    download_anthropic_hh(save_dir="data")
    dataset = Real_Toxicity_Prompt_Dataset()
    dataset.download_real_toxic_prompt(save_dir="data")
    toxic_samples = dataset.get_toxic_prompts(threshold=0.5, max_samples=10)
    dataset.save_toxic_samples_split(toxic_samples)