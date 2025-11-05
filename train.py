import time
import argparse
import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


import torch
import torch.nn as nn
import transformers
from model import LanguageModel
from torch.utils.data import Dataset, DataLoader
from transformers.optimization import AdamW, Adafactor
from transformers.trainer_pt_utils import get_parameter_names


from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

logger = get_logger(__name__)


class RegressionDataset(Dataset):
    def __init__(self, file, num_examples=-1):
        
        data = [json.loads(line) for line in open(file).readlines()]
        self.inputs = [instance["input"] for instance in data]
        self.outputs = [instance["output"] for instance in data]

        if num_examples != -1:
            self.inputs = self.inputs[:num_examples]
            self.outputs = self.outputs[:num_examples]
    
    def __len__(self):
        return len(self.inputs)

    def get_num_instances(self):
        return len(self.inputs)

    def __getitem__(self, index):
        s1, s2 = self.inputs[index], self.outputs[index]
        return s1, s2

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]

def configure_transformer_optimizer(model, args):
    "Prepare optimizer for transformer encoders"
    if args.optim == "adafactor":
        args.wd = 0

    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    # decay_parameters = [name for name in decay_parameters if "bias" not in name]
    decay_parameters = [name for name in decay_parameters if ("bias" not in name and 'scorer' not in name)]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": args.wd,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ] 
    
    if args.optim == "adamw":
        optimizer_kwargs = {"betas": (0.9, 0.999), "eps": 1e-8, "lr": args.learning_rate} 
        optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)
    
    elif args.optim == "adafactor":
        optimizer = Adafactor(optimizer_grouped_parameters, lr=args.learning_rate, scale_parameter=False, relative_step=False)
        
    return optimizer

    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Transformer Regression Model")
    parser.add_argument("--train_file", type=str, default="")
    parser.add_argument("--test_file", type=str, default="")
    parser.add_argument("--resume_from_weight", type=str, default="")
    parser.add_argument("--optim", type=str, default="adamw")
    parser.add_argument("--per_device_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--wd", type=float, default=0.01, help="Weight decay for AdamW.")
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--num_examples", type=int, default=-1)

    args = parser.parse_args()
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    accelerator.print(args)
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        exp_id = str(int(time.time()))
        vars(args)["exp_id"] = exp_id
        identifier = args.train_file.split("/")[-1].split("_")[1]
        args.output_dir = f"saved/{identifier}/{exp_id}/"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        args.write_file = f"{args.output_dir}/summary.txt"

        with open(args.write_file, "a") as f:
            f.write(str(args) + "\n\n")

        accelerator.project_configuration.automatic_checkpoint_naming = False

    accelerator.wait_for_everyone()

    with accelerator.main_process_first():
        train_dataset = RegressionDataset(args.train_file, args.num_examples)
        eval_dataset = RegressionDataset(args.test_file, args.num_examples)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.per_device_batch_size, collate_fn=train_dataset.collate_fn)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=args.per_device_batch_size, collate_fn=eval_dataset.collate_fn)
    
    model = LanguageModel()
    optimizer = configure_transformer_optimizer(model, args)
    model, optimizer = accelerator.prepare(model, optimizer)
    train_dataloader, eval_dataloader = accelerator.prepare(train_dataloader, eval_dataloader)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    total_batch_size = args.per_device_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)


    completed_steps = 0

    if args.resume_from_weight != "":
        weights = torch.load(args.resume_from_weight, map_location="cpu")
        model.load_state_dict(weights, strict=False)
        accelerator.print(f"Loaded pretrained weights from: {args.resume_from_weight}")

    device = accelerator.device
    loss_fn = nn.L1Loss().to(device)

    for epoch in range(args.num_train_epochs):
        total_loss, total_val_loss = 0, 0
        total_val_error, total_val_abs_error = 0, 0

        model.train()
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                inp, out = batch
                preds = model(inp)
                out = torch.tensor(out).to(device).unsqueeze(1)
                out = out/args.scale
                loss = loss_fn(preds, out)
                total_loss += loss.detach().float()
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        eval_progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)
        for step, batch in enumerate(eval_dataloader):
            with accelerator.accumulate(model) and torch.no_grad():
                inp, out = batch
                preds = model(inp)
                out = torch.tensor(out).to(device).unsqueeze(1)
                out = out / args.scale
                val_loss = loss_fn(preds, out)
                total_val_loss += val_loss.detach().float()
                eval_progress_bar.update(1)

                error = (preds - out).flatten().cpu().numpy()
                total_val_error += error.mean()
                total_val_abs_error += np.abs(error).mean()


        if accelerator.is_main_process:    
            result = {}
            result["epoch"] = epoch+1,
            result["step"] = completed_steps
            result["train_loss"] = round(total_loss.item()/len(train_dataloader), 4)
            result["val_loss"] = round(total_val_loss.item()/len(eval_dataloader), 4)
            result["val_error"] = round(total_val_error.item()/len(eval_dataloader), 4)
            result["val_absolute_error"] = round(total_val_abs_error.item()/len(eval_dataloader), 4)

            result_string = f"Epoch: {epoch}, Loss Train: {result['train_loss']}, Val: {result['val_loss']}; Val Error: {result['val_error']}, Absolute Error: {result['val_absolute_error']}\n"
            accelerator.print (result_string)

            with open(args.write_file, "a") as f:
                f.write(result_string + "\n")

            logger.info(result)

            unwrapped_model = accelerator.unwrap_model(model).to("cpu")
            state_dict = {name: params for name, params in unwrapped_model.named_parameters()}
            torch.save(state_dict, f"{args.output_dir}/epoch_{epoch+1}.pt")
            unwrapped_model = accelerator.unwrap_model(model).to(device)
