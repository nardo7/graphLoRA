from typing import Optional

import torch
from torch import nn
from transformers import GraphormerConfig, GraphormerForGraphClassification


class GraphormerRegularizedConfig(GraphormerConfig):
    def __init__(
        self,
        num_classes: int = 1,
        num_atoms: int = 512 * 9,
        num_edges: int = 512 * 3,
        num_in_degree: int = 512,
        num_out_degree: int = 512,
        num_spatial: int = 512,
        num_edge_dis: int = 128,
        multi_hop_max_dist: int = 5,
        spatial_pos_max: int = 1024,
        edge_type: str = "multi_hop",
        max_nodes: int = 512,
        share_input_output_embed: bool = False,
        num_hidden_layers: int = 12,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 768,
        num_attention_heads: int = 32,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0,
        encoder_normalize_before: bool = False,
        pre_layernorm: bool = False,
        apply_graphormer_init: bool = False,
        activation_fn: str = "gelu",
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        num_trans_layers_to_freeze: int = 0,
        traceable: bool = False,
        q_noise: float = 0,
        qn_block_size: int = 8,
        kdim: int = None,
        vdim: int = None,
        bias: bool = True,
        self_attention: bool = True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        for_finetuning: bool = True,
        beta_reg: float = 0.01,
        **kwargs,
    ):
        super().__init__(
            num_classes,
            num_atoms,
            num_edges,
            num_in_degree,
            num_out_degree,
            num_spatial,
            num_edge_dis,
            multi_hop_max_dist,
            spatial_pos_max,
            edge_type,
            max_nodes,
            share_input_output_embed,
            num_hidden_layers,
            embedding_dim,
            ffn_embedding_dim,
            num_attention_heads,
            dropout,
            attention_dropout,
            activation_dropout,
            layerdrop,
            encoder_normalize_before,
            pre_layernorm,
            apply_graphormer_init,
            activation_fn,
            embed_scale,
            freeze_embeddings,
            num_trans_layers_to_freeze,
            traceable,
            q_noise,
            qn_block_size,
            kdim,
            vdim,
            bias,
            self_attention,
            pad_token_id,
            bos_token_id,
            eos_token_id,
            **kwargs,
        )
        self.for_finetuning = for_finetuning
        self.beta_reg = beta_reg


class GraphormerClassifierRegularized(GraphormerForGraphClassification):
    config_class = GraphormerRegularizedConfig

    def __init__(self, config: GraphormerRegularizedConfig):
        super().__init__(config)
        self.config = config

    def post_init(self):
        super().post_init()
        # Initialize the pretrained_weights dict but don't populate it yet
        if self.config.for_finetuning:
            self.pretrained_weights: dict[str, torch.Tensor] = {}

    def _save_pretrained_weights(self):
        """Save the current model weights as pretrained weights for regularization."""
        if self.config.for_finetuning:
            for name, param in self.named_parameters():
                self.pretrained_weights[name] = param.detach().clone().to(self.device)
                self.pretrained_weights[name].requires_grad_(False)

    def forward(
        self,
        input_nodes: torch.LongTensor,
        input_edges: torch.LongTensor,
        attn_bias: torch.Tensor,
        in_degree: torch.LongTensor,
        out_degree: torch.LongTensor,
        spatial_pos: torch.LongTensor,
        attn_edge_type: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **unused,
    ):
        output = super().forward(
            input_nodes=input_nodes,
            input_edges=input_edges,
            attn_bias=attn_bias,
            in_degree=in_degree,
            out_degree=out_degree,
            spatial_pos=spatial_pos,
            attn_edge_type=attn_edge_type,
            labels=labels,
            return_dict=return_dict,
            **unused,
        )

        # reg_loss = self.regularization_loss()
        # head_los = output.loss if return_dict else output[0]
        # total_loss = head_los + self.config.beta_reg * reg_loss
        # if return_dict is None or return_dict:
        #     output.loss = total_loss
        # else:
        #     output[0] = total_loss
        return output

    def regularization_loss(self) -> torch.Tensor:
        weight_loss = torch.tensor(0.0, device=self.device)
        for name, param in self.named_parameters():
            if name in self.pretrained_weights:
                pretrained_param = self.pretrained_weights[name]
                if pretrained_param.device != param.device:
                    pretrained_param = pretrained_param.to(param.device)
                weight_loss += nn.functional.mse_loss(
                    param, pretrained_param, reduction="sum"
                )
        return weight_loss

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Load pretrained model and save the weights for regularization."""
        model = super().from_pretrained(*args, **kwargs)
        # Save pretrained weights after they're loaded
        if hasattr(model, "_save_pretrained_weights"):
            model._save_pretrained_weights()
        return model
