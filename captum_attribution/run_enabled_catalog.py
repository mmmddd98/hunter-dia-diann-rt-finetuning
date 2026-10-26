"""Run only explicitly enabled catalog entries, one reproducible config at a time."""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import yaml

DEFAULT_ROOT = Path("rt_model_evaluations")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--captum", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--gpu-workers",
        type=int,
        default=1,
        help="Independent GPU-bound runs (one GPU per worker; use at most 3 here).",
    )
    parser.add_argument(
        "--devices",
        help="Comma-separated GPUs for peptide parallelism within each run, e.g. cuda:0,cuda:1,cuda:2.",
    )
    args = parser.parse_args()
    if not 1 <= args.gpu_workers <= 3:
        raise SystemExit("--gpu-workers must be between 1 and 3")
    devices = (
        [item.strip() for item in args.devices.split(",") if item.strip()]
        if args.devices
        else []
    )
    if devices and args.gpu_workers != 1:
        raise SystemExit(
            "Use either --devices for peptide-parallel runs or --gpu-workers for model-parallel runs, not both."
        )
    if devices:
        import torch

        requested = []
        for device in devices:
            if (
                not device.startswith("cuda:")
                or not device.removeprefix("cuda:").isdigit()
            ):
                raise SystemExit(
                    "--devices accepts CUDA device names only, e.g. cuda:0,cuda:1,cuda:2"
                )
            requested.append(int(device.removeprefix("cuda:")))
        visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if max(requested) >= visible:
            raise SystemExit(
                f"The active Python interpreter sees {visible} CUDA device(s), but --devices requests "
                f"{','.join(devices)}. Activate your CUDA-enabled evaluation environment; "
                "also ensure CUDA_VISIBLE_DEVICES does not hide GPUs."
            )
    catalog = pd.read_csv(
        args.root / "registry/evaluation_catalog.tsv", sep="\t", keep_default_na=False
    )
    enabled = catalog[
        catalog.enabled.astype(str).str.lower().isin(["true", "1", "yes"])
    ]
    if enabled.empty:
        raise SystemExit(
            "No enabled rows. Review registry/evaluation_catalog.tsv and set enabled=true deliberately."
        )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "captum_attribution.materialize_catalog_configs",
            "--root",
            str(args.root),
        ],
        check=True,
    )
    commands = []
    for index, row in enumerate(enabled.itertuples(index=False)):
        config = (
            args.root
            / "registry/generated_configs"
            / f"{row.model_name}__{row.dataset_name}.yaml"
        )
        if devices:
            content = yaml.safe_load(config.read_text(encoding="utf-8"))
            content.setdefault("runtime", {})["devices"] = devices
            config.write_text(
                yaml.safe_dump(content, sort_keys=False), encoding="utf-8"
            )
        command = [
            sys.executable,
            "-m",
            "captum_attribution.evaluate_rt_model",
            "--config",
            str(config),
        ]
        command.append("--captum" if args.captum else "--prediction-only")
        if args.smoke_test:
            command.append("--smoke-test")
        commands.append((index % args.gpu_workers, command))

    def execute(gpu: int, command: list[str]) -> None:
        env = os.environ.copy()
        # Model-parallel jobs each receive one visible GPU.  In contrast, a
        # peptide-parallel job must retain every device requested through
        # --devices, otherwise cuda:1/cuda:2 become invalid logical IDs.
        if not devices:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        label = f"devices={','.join(devices)}" if devices else f"GPU {gpu}"
        print(f"{label}: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True, env=env)

    with ThreadPoolExecutor(max_workers=args.gpu_workers) as pool:
        futures = [pool.submit(execute, gpu, command) for gpu, command in commands]
        for future in as_completed(futures):
            future.result()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "captum_attribution.summarize_enabled_catalog",
            "--root",
            str(args.root),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
