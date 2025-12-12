import torch
from torch import nn
import math


class LoRALayer(nn.Module):
    def __init__(self, original_layer: nn.Linear, r: int = 4, alpha: int = 1):
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        self.original_layer.requires_grad_(False)  # Freeze original layer
        self.r = r
        self.alpha = alpha
        self.scaling = self.alpha / self.r

        # LoRA parameters
        self.lora_A = nn.Parameter(torch.zeros((r, original_layer.in_features)))
        self.lora_B = nn.Parameter(torch.zeros((original_layer.out_features, r)))

        # Initialize LoRA parameters
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        original_output = self.original_layer(x)
        lora_output = (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return original_output + lora_output


def apply_lora_to_model(
    model: nn.Module, module_names: list[str], r: int = 4, alpha: int = 1
):
    """
    Apply LoRA to specified linear layers in the model.

    Args:
        model: The model to modify.
        r: Rank of the LoRA decomposition.
        alpha: Scaling factor for LoRA.
        module_names: List of module names (as strings) to apply LoRA to.
    """
    applied_modules = []
    for name, module in model.named_modules():
        # looking for the specified module names
        if isinstance(module, nn.Linear) and any(mn in name for mn in module_names):
            lora_layer = LoRALayer(module, r=r, alpha=alpha)
            applied_modules.append(name)
            parent_module = model
            name_parts = name.split(".")
            # Traverse to the parent module
            for part in name_parts[:-1]:
                parent_module = getattr(parent_module, part)
            setattr(parent_module, name_parts[-1], lora_layer)

    print(f"Applied LoRA to modules: {applied_modules}")
