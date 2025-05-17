import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.modules.mamba2 import Mamba2

from functools import partial

class Mlp(nn.Module):
    def __init__(
        self,
        in_chs: int,
        hid_chs: int = None,
        out_chs: int = None,
        bias: bool = True,
        activation: nn.Module = nn.ReLU,
        norm: nn.Module = None,
        linear: nn.Module = nn.Linear,
        dropout: float = 0,
        depth: int = 2,
        device = None,
        dtype = None,
    ):
        super().__init__()
        from functools import partial
        factory_kwargs = {"device": device, "dtype": dtype}
        out_chs = out_chs if out_chs is not None else in_chs
        hid_chs = hid_chs if hid_chs is not None else in_chs
        dropout = partial(nn.Dropout, p=dropout) if dropout > 0 else None
        self.layers = [linear(in_chs, hid_chs if depth > 1 else out_chs, bias=bias, **factory_kwargs)]
        for _ in range(depth - 2):
            seq = [activation()]
            if norm is not None:
                seq.append(norm(hid_chs, **factory_kwargs))
            if dropout is not None:
                seq.append(dropout())
            seq.append(linear(hid_chs, hid_chs, bias=bias, **factory_kwargs))
            self.layers.append(nn.Sequential(*seq))
        if depth > 1:
            seq = [activation()]
            if norm is not None:
                seq.append(norm(hid_chs, **factory_kwargs))
            if dropout is not None:
                seq.append(dropout())
            seq.append(linear(hid_chs, out_chs, bias=bias, **factory_kwargs))
            self.layers.append(nn.Sequential(*seq))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x

class MPTSBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        block_norm: nn.Module = partial(nn.RMSNorm, eps=1e-6),
        mamba_d_state: int = 128,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_headdim: int = 64,
        mlp_expand: int = 4,
        mlp_bias: bool = True,
        mlp_activation: nn.Module = nn.SiLU,
        mlp_norm: nn.Module = None,
        mlp_dropout: float = 0,
        mlp_depth: int = 2,
        device = None,
        dtype = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.norm1 = block_norm(d_model, **factory_kwargs)
        self.token_mixer = Mamba2(
            d_model,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            headdim=mamba_headdim,
            use_mem_eff_path=True,
            **factory_kwargs
        )
        self.norm2 = block_norm(d_model, **factory_kwargs)
        self.channel_mixer = Mlp(
            d_model,
            hid_chs=int(d_model * mlp_expand),
            out_chs=d_model,
            bias=mlp_bias,
            activation=mlp_activation,
            norm=mlp_norm,
            dropout=mlp_dropout,
            depth=mlp_depth,
            **factory_kwargs
        )
    
    def forward(self, x):
        x = x + self.token_mixer(self.norm1(x))
        x = x + self.channel_mixer(self.norm2(x))
        return x


class MPTSModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_blocks: int = 1,
        block_norm: nn.Module = partial(nn.RMSNorm, eps=1e-6),
        mamba_d_state: int = 128,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_headdim: int = 64,
        mlp_expand: int = 4,
        mlp_bias: bool = True,
        mlp_activation: nn.Module = nn.SiLU,
        mlp_norm: nn.Module = None,
        mlp_dropout: float = 0,
        mlp_depth: int = 2,
        device = None,
        dtype = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.inproj = Mlp(in_chs=2, out_chs=d_model, bias=True, depth=1, **factory_kwargs)
        self.sos = nn.Parameter(torch.zeros(1, 1, d_model, **factory_kwargs))
        self.blocks = nn.ModuleList([
            MPTSBlock(
                d_model,
                block_norm=block_norm,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
                mamba_headdim=mamba_headdim,
                mlp_expand=mlp_expand,
                mlp_bias=mlp_bias,
                mlp_activation=mlp_activation,
                mlp_norm=mlp_norm,
                mlp_dropout=mlp_dropout,
                mlp_depth=mlp_depth,
                **factory_kwargs
            ) for _ in range(num_blocks)
        ])
        self.outproj = Mlp(in_chs=d_model, out_chs=2, bias=False, depth=1, **factory_kwargs)
    
    def forward(self, x):
        x = self.inproj(x)
        sos = self.sos.expand(x.shape[0], 1, -1)
        x = torch.cat([sos, x], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.outproj(x)
        return x

if __name__ == "__main__":
    model = MPTSModel(
        d_model=128,
        num_blocks=2,
        block_norm=partial(nn.RMSNorm, eps=1e-6),
        mamba_d_state=64,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_headdim=32,
        mlp_expand=4,
        mlp_bias=True,
        mlp_activation=nn.SiLU,
        mlp_norm=None,
        mlp_dropout=0,
        mlp_depth=2
    ).cuda()
    model.compile()
    print(sum(p.numel() for p in model.parameters()))
    x = torch.randn(32, 10, 2).cuda()
    y = model(x)
    print(y.shape)  # Should be (32, 10, 2)