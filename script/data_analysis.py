import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from utils.data_utils import Real_Toxicity_Prompt_Dataset
from utils.pred_utils import get_toxic_prediction

if __name__ == "__main__":
    dataset = Real_Toxicity_Prompt_Dataset()
    toxic_samples = dataset.get_toxic_prompts(threshold=0.9, max_samples=100)
