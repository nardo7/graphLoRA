import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers.models.graphormer.collating_graphormer import GraphormerDataCollator

from graphormer.classification import (
    GraphormerClassifierRegularized,
)
from simple_trainer import train

if __name__ == "__main__":
    # There is only one split on the hub
    dataset = load_dataset("OGB/ogbg-molhiv", cache_dir="./data")
    seed = 42
    # set seed for reproducibility in torch and numpy rng
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = dataset.shuffle(seed=seed)

    model: GraphormerClassifierRegularized = (
        GraphormerClassifierRegularized.from_pretrained(
            "clefourrier/pcqm4mv2_graphormer_base",
            num_classes=2,  # num_classes for the downstream task
            ignore_mismatched_sizes=True,
            beta_reg=0.01,
            for_finetuning=True,
        )
    )

    train_ds = dataset["train"].with_format("numpy")
    eval = dataset["validation"].with_format("numpy")

    train_dataloader = DataLoader(
        train_ds,
        batch_size=16,
        collate_fn=GraphormerDataCollator(on_the_fly_processing=True),
        num_workers=8,
        drop_last=True,
        prefetch_factor=2,
        shuffle=True,
    )
    eval_dataloader = DataLoader(
        eval,
        batch_size=16,
        collate_fn=GraphormerDataCollator(on_the_fly_processing=True),
        num_workers=8,
        drop_last=True,
        prefetch_factor=2,
        shuffle=False,
    )

    train(
        dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        model=model,
        device=torch.device("mps"),
        n_epochs=8,
        checkpoint_dir="./graph-classification/own_code/flag/molhiv",
        # resume_from_checkpoint=True,
        accumulate_gradient_steps=4,
    )
