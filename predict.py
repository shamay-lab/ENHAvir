import argparse
import numpy as np
import os
import pandas as pd
import torch
import transformers

from pathlib import Path
from tqdm.auto import tqdm
from model import LanguageModel
from torch.utils.data import Dataset, DataLoader


class RegressionDataset(Dataset):
    def __init__(self, data):
        
        self.inputs = [instance["input"] for instance in data]
        self.positions = [instance["position"] for instance in data]
        
    def __len__(self):
        return len(self.inputs)

    def get_num_instances(self):
        return len(self.inputs)

    def __getitem__(self, index):
        s1, s2 = self.inputs[index], self.positions[index]
        return s1, s2

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]


def get_eval_data(filename, seq_len, slide, batch_size):
    "Prepare dataloaders"
    
    sequence = [line.strip().lower() for line in open(filename).readlines()]
    sequence = "".join(sequence)
        
    data = []
    for k in tqdm(range(0, len(sequence), slide)):
        seq = sequence[k:k+seq_len]
        if len(seq) == seq_len:
            data.append({"input": " ".join([c for c in seq]), "position": k})
        
    eval_dataset = RegressionDataset(data)
    eval_loader = DataLoader(eval_dataset, shuffle=False, batch_size=batch_size, collate_fn=eval_dataset.collate_fn)        
    eval_positions = [line["position"] for line in data]
    return eval_loader, data, eval_positions


def predict_from_lm(model, dataloader, scale):
    all_preds = []
    for inp, _ in tqdm(dataloader, leave=False):
        with torch.no_grad():
            preds = model(inp)
        all_preds += list(preds.detach().flatten().cpu().numpy())
    all_preds = [round(item * scale, 4) for item in all_preds]
    return all_preds


def postprocess(values, positions, window_size=4, satisfy_percent=25, threshold_percentile=80):
    new_values = []
    max_position = len(positions)
    threshold = np.percentile(values, threshold_percentile)
    
    for start_index in range(len(values)):
        satisfy = []
        for window_index in range(window_size):            
            window_start = start_index + window_index + 1 - window_size
            window_end = start_index + window_index
            
            if window_start >= 0 and window_end < max_position:
                if max(values[window_start:window_end+1]) > threshold:
                    satisfy.append(True)
                else:
                    satisfy.append(False)

        if satisfy.count(True) / len(satisfy) >= satisfy_percent / 100:
            new_values.append(max(0, values[start_index] - threshold))
        else:
            new_values.append(0)

    new_values = [round(item, 4) for item in new_values]
    return new_values
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Transformer Regression Model")
    parser.add_argument("--seq_len", type=int, default=200, help="DNA sequence length of each window for prediction.")
    parser.add_argument("--slide", type=int, default=50, help="Sliding length for prediction.")
    parser.add_argument("--window", type=int, default=4, help="Window length for post-processing.")
    parser.add_argument("--satisfy", type=int, default=25, help="Satisfaction % for ")
    parser.add_argument("--threshold", type=int, default=80, help="Threshold percentile for ")
    parser.add_argument("--filename", type=str, default="", help="File containing sequence to run prediction on.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation.")
    parser.add_argument("--scale", type=int, default=76, help="Scaling factor to de-normalize the predictions. Value chosen based on the training data. Recommended to keep the value unchanged at 76.")
    parser.add_argument("--outfilename", type=str, default="", help="File to write outputs.")

    
    args = parser.parse_args()
    print(args)
    
    transformers.utils.logging.set_verbosity_error()
    
    model = LanguageModel()
    model.cuda()
    model.eval()

    print ("Loading pretrained weights.")
    weights = torch.load("weights/checkpoint.pt", map_location="cpu")
    model.load_state_dict(weights, strict=False)

    file_identifier = args.filename[:-4].split("/")[-1]

    print ("Loading data and running predictions.")
    eval_loader, eval_data, eval_positions = get_eval_data(
        args.filename, args.seq_len, args.slide, args.batch_size
    )
    print (f"Total evaluation samples: {len(eval_data)}")
    eval_preds = predict_from_lm(model, eval_loader, args.scale)
    final_preds = postprocess(
        eval_preds, eval_positions, args.window, args.satisfy, args.threshold
    )
    eval_result_df = pd.DataFrame({
        "Position": eval_positions, "Predicted Value": final_preds
    })

    if not os.path.exists("results"):
        os.makedirs("results")

    if args.outfilename == "":
        out_file_name = f"results/{file_identifier}_predictions.csv"
    else:
        out_file_name = f"results/{args.outfilename}"

    eval_result_df.to_csv(out_file_name, index=False)
    print (f'Saved results in {out_file_name}')
    