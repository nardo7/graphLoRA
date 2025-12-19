from dataclasses import dataclass
from typing import Mapping, Optional, Union, cast

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from transformers.modeling_outputs import SequenceClassifierOutput

from checkpointing import find_last_checkpoint, load_checkpoint, save_checkpoint
from optim import get_optimizer, get_scheduler


@dataclass
class FLAGConfig:
    # default from the official graphormer code
    use_FLAG: bool = False
    m: int = 3
    step_size: float = 0.01
    epsilon: float = 0


def train(
    dataloader: torch.utils.data.DataLoader,
    eval_dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    n_epochs,
    device=torch.device("cpu"),
    checkpoint_dir: Optional[str] = None,
    resume_from_checkpoint: bool = False,
    accumulate_gradient_steps: int = 1,
    flag_config: FLAGConfig = FLAGConfig(),
):
    # Initialize optimizer and scheduler
    num_steps = n_epochs * len(dataloader)
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer, num_steps)
    current_epoch = 0
    batch_size: int = cast(int, dataloader.batch_size)

    # Load from checkpoint if desired (model, optimizer and scheduler states)
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

    # training loop
    for epoch in range(current_epoch, n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_steps = 0
        accumulated_steps = 0

        # Calculate total steps based on simulated batch size
        total_gradient_steps = len(dataloader) // (accumulate_gradient_steps) + (
            1 if len(dataloader) % accumulate_gradient_steps > 0 else 0
        )

        # Create progress bar with simulated batch steps
        pbar = tqdm(
            total=total_gradient_steps,
            desc=f"Epoch {epoch + 1}/{n_epochs} - Training Loss: {train_loss:.4f}",
            leave=False,
        )

        for step, data in enumerate(dataloader):
            input = _prepare_input(data, device)
            input = cast(dict[str, torch.Tensor], input)

            if flag_config.use_FLAG:
                train_loss += _flag_inner_loop(
                    model,
                    input,
                    flag_config.m,
                    flag_config.step_size,
                    flag_config.epsilon,
                )
            else:
                output: SequenceClassifierOutput = model(**input)

                loss = output.loss

                # Normalize loss by accumulation steps
                loss = loss / accumulate_gradient_steps

                # apply regularization loss if applicable only once when the gradients will be applied instead of every step, which give the same loss
                if can_apply_gradient_step(
                    accumulated_steps + 1,
                    accumulate_gradient_steps,
                    step,
                    len(dataloader),
                ) and hasattr(model, "regularization_loss"):
                    loss += model.config.beta_reg * model.regularization_loss()

                loss.backward()

                train_loss += loss.detach().item() * accumulate_gradient_steps
            accumulated_steps += 1

            # Only update weights after accumulating enough gradients
            if can_apply_gradient_step(
                accumulated_steps, accumulate_gradient_steps, step, len(dataloader)
            ):
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                train_steps += 1
                accumulated_steps = 0
                # Update progress bar only on actual gradient steps
                pbar.update(1)

        pbar.close()

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


def can_apply_gradient_step(
    accumulated_steps: int,
    step_size: int,
    step: int,
    total_steps: int,
):
    return accumulated_steps >= step_size or (step + 1) == total_steps


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
        for batch, data in tqdm(
            enumerate(dataloader), desc="Evaluating", total=len(dataloader)
        ):
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


def _flag_inner_loop(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    m: int,
    step_size: float,
    epsilon: float,
    loss_fn: nn.Module = nn.CrossEntropyLoss(),
) -> float:
    # return batch
    train_loss = 0.0
    x = batch["input_nodes"]
    perturb = (
        nn.init.uniform_(torch.FloatTensor(*x.shape), -step_size, step_size)
        .to(x.device)
        .requires_grad_(True)
    )
    for _ in range(m):
        x = batch["input_nodes"]
        x = x + perturb
        batch["input_nodes"] = x

        out = model(**batch)
        loss = out.loss / m
        loss.backward()  # accumulates model grads + perturb.grad
        with torch.no_grad():
            perturb += step_size * perturb.grad.sign()
        perturb.grad.zero_()  # ONLY reset perturb grad
        train_loss += loss.detach().item()
    return train_loss


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
