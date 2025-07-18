import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from utils.data_utils import download_anthropic_hh

if __name__ == "__main__":
    download_anthropic_hh(save_dir="data")