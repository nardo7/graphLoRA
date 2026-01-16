from functools import partial
import math

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR


def get_optimizer(opt_model: nn.Module, learning_rate: float = 2e-4):
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
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    print("Optimizer created with learning rate:", learning_rate)
    return optimizer


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, num_warmup_steps: int, num_training_steps: int
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def _get_linear_schedule_with_warmup_lr_lambda(
    current_step: int, num_warmup_steps: int, num_training_steps: int
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0,
        float(num_training_steps - current_step)
        / float(max(1, num_training_steps - num_warmup_steps)),
    )


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    num_train_steps,
    num_warmup_steps=None,
    last_epoch=-1,
):
    # Default warmup: 20% of training steps for stability
    if num_warmup_steps is None:
        num_warmup_steps = max(1, int(0.2 * num_train_steps))

    # build the cosine scheduler with warmup
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_linear_scheduler(
    optimizer: torch.optim.Optimizer,
    num_train_steps,
    num_warmup_steps=None,
    last_epoch=-1,
):
    # Default warmup: 10% of training steps
    if num_warmup_steps is None:
        num_warmup_steps = max(1, int(0.1 * num_train_steps))

    # build the linear scheduler with warmup
    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
