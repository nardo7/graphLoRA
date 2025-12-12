from functools import partial
from typing import Mapping, Union, cast, Optional
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
import math
import os
import glob
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import accuracy_score, roc_auc_score


class TrainingConfig:
    pass


def train(
    dataloader: torch.utils.data.DataLoader,
    eval_dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    n_epochs,
    device=torch.device("cpu"),
    checkpoint_dir: Optional[str] = None,
    resume_from_checkpoint: bool = False,
    accumulate_gradient_steps: int = 1,
):
    num_steps = n_epochs * len(dataloader)
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer, num_steps)
    current_epoch = 0
    if resume_from_checkpoint and checkpoint_dir is not None:
        path = find_last_checkpoint(checkpoint_dir)
        model, optimizer, scheduler, training_state = load_checkpoint(
            model=model,
            checkpoint_dir=checkpoint_dir,
            checkpoint_path=path,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        current_epoch = training_state["epoch"]
        num_steps = (n_epochs * len(dataloader)) - (current_epoch * len(dataloader))
    else:
        model.to(device)

    # Track best ROC-AUC for checkpointing
    best_roc_auc = (
        training_state.get("best_roc_auc", -1.0) if resume_from_checkpoint else -1.0
    )

    for epoch in range(current_epoch, n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_steps = 0
        accumulated_steps = 0

        for step, data in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch + 1}/{n_epochs} - Training")
        ):
            input = _prepare_input(data, device)
            input = cast(dict[str, torch.Tensor], input)

            output: SequenceClassifierOutput = model(**input)

            loss = output.loss
            # Normalize loss by accumulation steps
            loss = loss / accumulate_gradient_steps
            loss.backward()

            train_loss += loss.detach().item() * accumulate_gradient_steps
            accumulated_steps += 1

            # Only update weights after accumulating enough gradients
            if accumulated_steps >= accumulate_gradient_steps or (step + 1) == len(
                dataloader
            ):
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                train_steps += 1
                accumulated_steps = 0

        avg_train_loss = train_loss / train_steps
        print(
            f"Epoch {epoch + 1}/{n_epochs} - Average Training Loss: {avg_train_loss:.4f}"
        )

        # Evaluation phase
        if eval_dataloader is not None:
            eval_metrics = evaluate(model, eval_dataloader, device)
            print(f"Epoch {epoch + 1}/{n_epochs} - Evaluation Metrics: {eval_metrics}")

            # Checkpointing based on ROC-AUC improvement
            if checkpoint_dir is not None and "roc_auc" in eval_metrics:
                current_roc_auc = eval_metrics["roc_auc"]
                if current_roc_auc > best_roc_auc:
                    best_roc_auc = current_roc_auc
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        metrics=eval_metrics,
                        checkpoint_dir=checkpoint_dir,
                    )
                    print(f"✓ Checkpoint saved! New best ROC-AUC: {best_roc_auc:.4f}")


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

    # Load model
    model = model.from_pretrained(checkpoint_path)
    model.to(device)

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


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device=torch.device("cpu"),
):
    """
    Evaluate the model on a given dataloader.
    Returns a dictionary with evaluation metrics.
    """
    model.eval()
    total_loss = 0.0
    num_steps = 0

    # First pass: collect all data to determine sizes
    batch_size: int = cast(int, dataloader.batch_size)
    all_predictions = np.zeros((len(dataloader), batch_size), dtype=int)
    all_labels = np.zeros((len(dataloader), batch_size), dtype=int)
    all_logits = np.zeros(
        (len(dataloader), batch_size, model.config.num_classes), dtype=float
    )

    with torch.no_grad():
        for batch, data in tqdm(enumerate(dataloader), desc="Evaluating"):
            input = _prepare_input(data, device)
            input = cast(dict[str, torch.Tensor], input)

            labels: torch.Tensor = input.get("labels")
            output: SequenceClassifierOutput = model(**input)

            if output.loss is not None:
                total_loss += output.loss.item()
            num_steps += 1

            # Get predictions
            logits = output.logits.cpu().numpy()
            predictions = np.argmax(logits, axis=-1)

            # Accumulate results
            all_labels[batch, :] = labels.cpu().numpy()
            all_predictions[batch, :] = predictions
            all_logits[batch, :, :] = logits

    # Calculate metrics
    metrics = {}
    metrics["eval_loss"] = total_loss / num_steps if num_steps > 0 else 0.0

    all_labels = all_labels.flatten()
    all_predictions = all_predictions.flatten()
    if all_labels is not None:
        # Accuracy
        metrics["accuracy"] = accuracy_score(all_labels, all_predictions)

        # ROC-AUC (for binary classification)
        if all_logits.shape[-1] == 2:
            # Use probabilities for positive class
            metrics["roc_auc"] = roc_auc_score(all_labels, all_predictions)

    return metrics


def get_optimizer(opt_model: nn.Module):
    # decay_parameters = [
    #     name
    #     for name, param in opt_model.named_parameters()
    #     if not any(nd in name for nd in ["bias", "LayerNorm.weight"])
    # ]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in opt_model.named_parameters() if (p.requires_grad)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=5e-5)
    return optimizer


def _get_linear_schedule_with_warmup_lr_lambda(
    current_step: int, num_training_steps: int
):
    return max(
        0.0,
        float(num_training_steps - current_step) / float(max(1, num_training_steps)),
    )


def get_scheduler(optimizer: torch.optim.Optimizer, num_train_steps, last_epoch=-1):
    # build the lineal scheduler
    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_training_steps=num_train_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _prepare_input(
    data: Union[torch.Tensor, dict], device: torch.device
) -> Union[torch.Tensor, dict[str, torch.Tensor], list[torch.Tensor]]:
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: _prepare_input(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(_prepare_input(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    return data
