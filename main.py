from train import train
import argparse
from data import get_cleaned_dataset
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--remove_outliers', type=int, default=0)
    args = parser.parse_args()

    if os.path.exists(args.model_path) == False:
        os.makedirs(args.model_path)
    
    df = get_cleaned_dataset("perth-house-prices/all_perth_310121.csv", encoder_path=args.model_path, args = args)
    model = train(args, df)
