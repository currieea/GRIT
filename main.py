import argparse
import importlib.metadata
import json
import os
import platform
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda
import wandb

from solver import ECMP, ERM, Fish, GroupDRO, IRM, LISA, MatchDG, REx, SWAD


SOLVERS = {
    solver.__name__: solver
    for solver in (ERM, IRM, REx, Fish, GroupDRO, ECMP, MatchDG, LISA, SWAD)
}


def build_parser():
    parser = argparse.ArgumentParser(description="GRIT domain generalization experiments")
    parser.add_argument("--no_wandb", default=False, action="store_true")
    parser.add_argument("--wandb_project", default="CMP")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--root_dir", type=Path, default=Path("data"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--deterministic", default=False, action="store_true")
    parser.add_argument("--seed", default=1001, type=int)
    parser.add_argument("--dataset", type=str, default="ColoredMNIST")
    parser.add_argument("--latent_dim", default=512, type=int)
    parser.add_argument("--pretrained", default="false", choices=("true", "false"))
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--split_scheme", default="official", type=str)
    parser.add_argument("--solver", choices=tuple(SOLVERS), default="ERM")
    parser.add_argument("--param1", default=100, type=float)
    parser.add_argument("--param2", default=0, type=float)
    parser.add_argument("--param3", default=0, type=float)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--upweighting", default="false", choices=("true", "false"))
    parser.add_argument("--featurizer", default="linear", type=str)
    parser.add_argument("--projection", default="oracle", type=str)
    return parser


def resolve_device(requested):
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed, deterministic=False):
    """Seed all random number generators used by the training code."""
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _dataset_manifest(dataset):
    try:
        manifest_path = Path(dataset.data_dir) / "reproduction_manifest.json"
    except (AttributeError, TypeError):
        return None
    if not manifest_path.is_file():
        return None
    with manifest_path.open() as handle:
        return json.load(handle)


def _installed_version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _atomic_json_dump(payload, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=str(output_path.parent), prefix=output_path.name, suffix=".tmp", delete=False
    ) as handle:
        json.dump(_json_value(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(str(temporary_path), str(output_path))


def build_result(args, solver, elapsed_seconds):
    config = vars(args).copy()
    config.pop("output_json", None)
    return {
        "schema_version": 1,
        "status": "completed",
        "config": _json_value(config),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": _installed_version("torchvision"),
            "numpy": _installed_version("numpy"),
            "wilds": _installed_version("wilds"),
            "wandb": _installed_version("wandb"),
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": str(solver.device),
        },
        "dataset_manifest": _dataset_manifest(solver.dataset),
        "selection": {
            "in_domain": {
                "split": "in_test",
                "step": getattr(solver, "best_id_step", None),
                "metrics": solver.best_id_log,
            },
            "validation": {
                "split": "val",
                "step": getattr(solver, "best_val_step", None),
                "metrics": solver.best_val_log,
            },
            "oracle_test": {
                "split": "test",
                "step": getattr(solver, "best_oracle_step", None),
                "metrics": solver.best_oracle_log,
            },
        },
        "elapsed_seconds": elapsed_seconds,
    }


def main(args):
    args.root_dir = args.root_dir.expanduser().resolve()
    device = resolve_device(args.device)
    set_seed(args.seed, deterministic=args.deterministic)

    hparam = vars(args).copy()
    hparam["root_dir"] = str(args.root_dir)
    hparam["device"] = device
    hparam["wandb"] = not args.no_wandb

    if hparam["wandb"]:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            config={key: _json_value(value) for key, value in hparam.items() if key != "device"},
        )
        wandb.run.log_code()

    started = time.monotonic()
    solver = SOLVERS[hparam["solver"]](hparam)
    solver.fit()
    result = build_result(args, solver, time.monotonic() - started)
    if args.output_json is not None:
        _atomic_json_dump(result, args.output_json)
    return result


if __name__ == "__main__":
    main(build_parser().parse_args())
