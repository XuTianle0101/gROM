import argparse
import json
import yaml
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class TrainConfig:
    latent_size_gnn: int = 16
    latent_size_mlp: int = 64
    process_iterations: int = 3
    number_hidden_layers_mlp: int = 2
    learning_rate: float = 0.001
    batch_size: int = 100
    lr_decay: float = 0.001
    nepochs: int = 100
    weight_decay: float = 1e-5
    rate_noise: float = 100
    rate_noise_features: float = 1e-5
    stride: int = 5
    bcs_gnn: str = 'models_bcs/31.10.2022_01.35.31'
    label_norm: int = 1

    @staticmethod
    def from_args() -> "TrainConfig":
        p = argparse.ArgumentParser(description='Graph Reduced Order Models')
        p.add_argument('--config', type=str, default=None)
        p.add_argument('--bs', type=int, default=100)
        p.add_argument('--epochs', type=int, default=100)
        p.add_argument('--lr_decay', type=float, default=0.001)
        p.add_argument('--lr', type=float, default=0.001)
        p.add_argument('--rate_noise', type=float, default=100)
        p.add_argument('--rate_noise_features', type=float, default=1e-5)
        p.add_argument('--weight_decay', type=float, default=1e-5)
        p.add_argument('--ls_gnn', type=int, default=16)
        p.add_argument('--ls_mlp', type=int, default=64)
        p.add_argument('--process_iterations', type=int, default=3)
        p.add_argument('--hl_mlp', type=int, default=2)
        p.add_argument('--label_norm', type=int, default=1)
        p.add_argument('--stride', type=int, default=5)
        p.add_argument('--bcs_gnn', type=str, default='models_bcs/31.10.2022_01.35.31')
        args = p.parse_args()
        if args.config:
            cfg = yaml.safe_load(Path(args.config).read_text())
            return TrainConfig(**cfg)
        return TrainConfig(
            latent_size_gnn=args.ls_gnn,
            latent_size_mlp=args.ls_mlp,
            process_iterations=args.process_iterations,
            number_hidden_layers_mlp=args.hl_mlp,
            learning_rate=args.lr,
            batch_size=args.bs,
            lr_decay=args.lr_decay,
            nepochs=args.epochs,
            weight_decay=args.weight_decay,
            rate_noise=args.rate_noise,
            rate_noise_features=args.rate_noise_features,
            stride=args.stride,
            bcs_gnn=args.bcs_gnn,
            label_norm=args.label_norm,
        )

    def to_legacy_params(self):
        d = asdict(self)
        d.pop('label_norm')
        return d
