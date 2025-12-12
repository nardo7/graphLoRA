import math
import os
import glob
import torch
from torch import nn
from typing import Optional
from safetensors.torch import load_file


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metrics: dict,
    checkpoint_dir: str,
):
    """
    Save model checkpoint with optimizer and scheduler state.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-epoch-{epoch + 1}")
    os.makedirs(checkpoint_path, exist_ok=True)

    # Save model
    model.save_pretrained(checkpoint_path)

    # Save optimizer state
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))

    # Save scheduler state
    torch.save(scheduler.state_dict(), os.path.join(checkpoint_path, "scheduler.pt"))

    # Save training state (epoch, metrics, and best metric)
    training_state = {
        "epoch": epoch + 1,
        "metrics": metrics,
        "best_roc_auc": metrics.get("roc_auc", 0.0),
        # "optimizer_steps": optimizer.state_dict()["state"].get(0, {}).get("step", 0)
        # if optimizer.state_dict()["state"]
        # else 0,
    }
    torch.save(training_state, os.path.join(checkpoint_path, "trainer_state.pt"))

    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    model: nn.Module,
    checkpoint_dir: str,
    checkpoint_path: Optional[str] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR, dict]:
    """
    Load model checkpoint with optimizer and scheduler state.

    Args:
        model: The model to load weights into
        checkpoint_dir: Base directory containing checkpoints
        checkpoint_path: Specific checkpoint path to load. If None, loads the best checkpoint based on ROC-AUC
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load the model onto

    Returns:
        tuple: (model, optimizer, scheduler, training_state)
    """
    # Find the best checkpoint if no specific path is given
    if checkpoint_path is None:
        checkpoint_path = find_best_checkpoint(checkpoint_dir)
        if checkpoint_path is None:
            raise ValueError(f"No checkpoints found in {checkpoint_dir}")
        print(f"Loading best checkpoint: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

    # Load model weights into the existing model (which should already have LoRA applied)
    # Use load_state_dict with strict=False to allow missing keys (like LoRA params if they don't match)

    model_file = os.path.join(checkpoint_path, "model.safetensors")
    if os.path.exists(model_file):
        state_dict = load_file(model_file, device=str(device))
    else:
        # Fallback to pytorch_model.bin if safetensors doesn't exist
        model_file = os.path.join(checkpoint_path, "pytorch_model.bin")
        state_dict = torch.load(model_file, map_location=device)

    model.load_state_dict(state_dict)
    model.to(device)
    print(f"Model weights loaded from {checkpoint_path}")

    # Load optimizer state if provided
    optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
    if optimizer is not None and os.path.exists(optimizer_path):
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
        print("Optimizer state loaded")

    # Load scheduler state if provided
    scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
    if scheduler is not None and os.path.exists(scheduler_path):
        scheduler.load_state_dict(torch.load(scheduler_path, map_location=device))
        print("Scheduler state loaded")

    # Load training state
    trainer_state_path = os.path.join(checkpoint_path, "trainer_state.pt")
    training_state = None
    if os.path.exists(trainer_state_path):
        training_state = torch.load(trainer_state_path, map_location=device)
        print(
            f"Training state loaded: Epoch {training_state['epoch']}, "
            f"Best ROC-AUC: {training_state.get('best_roc_auc', 'N/A')}"
        )

    return model, optimizer, scheduler, training_state


def find_last_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the last checkpoint based on epoch number.

    Args:
        checkpoint_dir: Directory containing checkpoint subdirectories

    Returns:
        Path to the last checkpoint, or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None

    # Find all checkpoint directories
    checkpoint_dirs = glob.glob(os.path.join(checkpoint_dir, "checkpoint-epoch-*"))

    if not checkpoint_dirs:
        return None

    # Sort checkpoints by epoch number
    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]), reverse=True)

    # Return the latest checkpoint
    return checkpoint_dirs[0]


def find_best_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the checkpoint with the best ROC-AUC score.

    Args:
        checkpoint_dir: Directory containing checkpoint subdirectories

    Returns:
        Path to the best checkpoint, or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None

    # Find all checkpoint directories
    checkpoint_dirs = glob.glob(os.path.join(checkpoint_dir, "checkpoint-epoch-*"))

    if not checkpoint_dirs:
        return None

    best_checkpoint = None
    best_roc_auc = -1.0

    # Iterate through checkpoints and find the one with best ROC-AUC
    for ckpt_dir in checkpoint_dirs:
        trainer_state_path = os.path.join(ckpt_dir, "trainer_state.pt")
        if os.path.exists(trainer_state_path):
            training_state = torch.load(trainer_state_path, map_location="cpu")
            roc_auc = training_state.get("best_roc_auc", -1.0)

            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_checkpoint = ckpt_dir

    return best_checkpoint
