from functools import partial
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
import math


class TrainingConfig:
    pass


def train(
    dataloader,
    model,
    n_epochs,
    optimizer=None,
    loss_fn=nn.CrossEntropyLoss(),
    device=torch.device("cpu"),
):
    if optimizer is None:
        optimizer = get_optimizer(model)
        scheduler = get_scheduler(optimizer, 0)
    model.train()
    for epoch in tqdm(range(n_epochs)):
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
        # checkpointing given metrics

        print("Epoch {}, loss: {}".format(epoch + 1, loss.item()))


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
