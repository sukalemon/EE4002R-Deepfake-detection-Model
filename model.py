import timm
import torch
import torch.nn as nn
from timm.models.pvt_v2 import PyramidVisionTransformerStage


def get_best_coatnet_name() -> str:
    coatnet_candidates = [
        "coatnet_0_rw_224.sw_in1k",
        "coatnet_0_rw_224",
        "coatnet_1_rw_224.sw_in1k",
        "coatnet_1_rw_224",
        "coatnet_2_rw_224.sw_in12k_ft_in1k",
        "coatnet_2_rw_224",
    ]
    available = set(timm.list_models())
    for name in coatnet_candidates:
        if name in available:
            return name
    raise ValueError("No supported CoAtNet model found in this timm install.")


def choose_num_heads(channels: int) -> int:
    for h in [24, 16, 12, 8, 6, 4, 2]:
        if channels % h == 0:
            return h
    return 1


class SingleBranch(nn.Module):
    def __init__(self, coatnet_name: str, activation_type: str, dropout: float = 0.4, elu_alpha: float = 1.0):
        super().__init__()

        self.cnn = timm.create_model(
            coatnet_name,
            pretrained=True,
            features_only=True,
            out_indices=(3,),
        )

        channels = self.cnn.feature_info.channels()[-1]
        num_heads = choose_num_heads(channels)

        self.pvt_stage = PyramidVisionTransformerStage(
            dim=channels,
            dim_out=channels,
            depth=1,
            downsample=False,
            num_heads=num_heads,
            sr_ratio=1,
            linear_attn=False,
            mlp_ratio=4.0,
            qkv_bias=True,
            proj_drop=0.0,
            attn_drop=0.0,
            drop_path=0.1,
            norm_layer=nn.LayerNorm,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        if activation_type.lower() == "gelu":
            self.activation = nn.GELU()
            head_act = nn.GELU()
        elif activation_type.lower() == "elu":
            self.activation = nn.ELU(alpha=elu_alpha)
            head_act = nn.ELU(alpha=elu_alpha)
        else:
            raise ValueError("activation_type must be 'gelu' or 'elu'")

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels, 128),
            head_act,
            nn.Dropout(dropout),
        )

        self.out_dim = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)[0]           # [B, C, H, W]
        feat = feat.permute(0, 2, 3, 1) # [B, H, W, C]
        feat = self.pvt_stage(feat)     # [B, C, H, W]
        feat = self.activation(feat)
        feat = self.pool(feat).flatten(1)
        feat = self.head(feat)
        return feat


class DualBranchCoAtNetPVTv2Classifier(nn.Module):
    def __init__(self, dropout: float = 0.4, elu_alpha: float = 1.0):
        super().__init__()

        coatnet_name = get_best_coatnet_name()

        self.gelu_branch = SingleBranch(
            coatnet_name=coatnet_name,
            activation_type="gelu",
            dropout=dropout,
            elu_alpha=elu_alpha,
        )
        self.elu_branch = SingleBranch(
            coatnet_name=coatnet_name,
            activation_type="elu",
            dropout=dropout,
            elu_alpha=elu_alpha,
        )

        self.gelu_norm = nn.LayerNorm(self.gelu_branch.out_dim)
        self.elu_norm = nn.LayerNorm(self.elu_branch.out_dim)

        self.gelu_weight = nn.Parameter(torch.tensor(1.0))
        self.elu_weight = nn.Parameter(torch.tensor(1.0))

        fused_dim = self.gelu_branch.out_dim + self.elu_branch.out_dim

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gelu_feat = self.gelu_branch(x)
        elu_feat = self.elu_branch(x)

        gelu_feat = self.gelu_norm(gelu_feat)
        elu_feat = self.elu_norm(elu_feat)

        gelu_feat = self.gelu_weight * gelu_feat
        elu_feat = self.elu_weight * elu_feat

        fused = torch.cat([gelu_feat, elu_feat], dim=1)
        logits = self.classifier(fused)
        return logits
