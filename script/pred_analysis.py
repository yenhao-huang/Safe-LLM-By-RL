import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from utils.pred_utils import get_toxic_prediction

if __name__ == "__main__":
    get_toxic_prediction(path="../predicts/predictions.jsonl")