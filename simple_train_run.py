import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GraphormerForGraphClassification
from transformers.models.graphormer.collating_graphormer import GraphormerDataCollator

from lora import apply_lora_to_model
from simple_trainer import train

if __name__ == "__main__":
    # There is only one split on the hub
    dataset = load_dataset("OGB/ogbg-molhiv", cache_dir="./data")
    seed = 42
    dataset = dataset.shuffle(seed=seed)
    apply_lora = False
    checkpoint_dir = "./graph-classification/own_code/full_b_64/molhiv"
    resume_from_checkpoint = False
    lr = 2e-4  # use 2e-4 for full fine-tuning, 2e-3 for LoRA

    model: GraphormerForGraphClassification = (
        GraphormerForGraphClassification.from_pretrained(
            "clefourrier/pcqm4mv2_graphormer_base",
            num_classes=2,  # num_classes for the downstream task
            ignore_mismatched_sizes=True,
        )
    )

    if apply_lora:
        # Apply LoRA to specific modules
        apply_lora_to_model(
            model,
            module_names=["q_proj", "k_proj", "v_proj", "lm_head_transform_weight"],
            r=16,
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
        checkpoint_dir=checkpoint_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        accumulate_gradient_steps=4,  # simulating batch size of 64
        learning_rate=lr,
        max_grad_norm=10.0,  # Higher threshold for 2e-4 LR
        num_warmup_steps=None,  # Defaults to 10% of total steps
    )
