#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List, Tuple

from plot_prefix_confusion import build_progression, save_plot


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_case_path(source_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return source_dir / path


def load_cases(source_dir: Path) -> List[Tuple[str, Path]]:
    manifest_path = source_dir / "manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        cases = []
        for case in payload.get("cases", []):
            name = case["name"]
            gnf_path = _resolve_case_path(source_dir, case["gnf_path"])
            cases.append((name, gnf_path))
        if cases:
            return cases

    cases = []
    for gnf_path in sorted(source_dir.glob("*_gnf.json")):
        name = gnf_path.stem.replace("_gnf", "")
        cases.append((name, gnf_path))
    if not cases:
        raise ValueError(f"No manifest cases or *_gnf.json files found in {source_dir}")
    return cases


def emit_case(
    checkpoint: str,
    device: str,
    out_dir: Path,
    name: str,
    gnf_payload: dict,
    topk: int,
) -> None:
    gnf_path = out_dir / f"{name}_gnf.json"
    json_path = out_dir / f"{name}_confusion.json"
    entropy_png_path = out_dir / f"{name}_entropy_confusion.png"
    xent_png_path = out_dir / f"{name}_target_cross_entropy.png"

    write_json(gnf_path, gnf_payload)
    result = build_progression(
        checkpoint_path=checkpoint,
        device_arg=device,
        d=int(gnf_payload["d"]),
        factors=[tuple(factor) for factor in gnf_payload["gnf_factors"]],
        topk=topk,
        truncate_overflow=False,
    )
    result["source_gnf_path"] = str(gnf_payload.get("source_gnf_path", gnf_path))
    write_json(json_path, result)

    title_base = name.replace("_", " ").title()
    save_plot(
        result["progression"],
        str(entropy_png_path),
        f"{title_base} Entropy Confusion",
        metric_key="entropy_confusion_score",
        y_label="Entropy Confusion",
    )
    save_plot(
        result["progression"],
        str(xent_png_path),
        f"{title_base} Target Cross-Entropy",
        metric_key="target_cross_entropy",
        y_label="Target Cross-Entropy",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rescore a saved GWY-kernel/random GNF suite with a new checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt or snapshot")
    parser.add_argument("--device", default="cuda", help="cuda, cpu, cuda:0, ...")
    parser.add_argument(
        "--source-suite-dir",
        required=True,
        help="Directory containing manifest.json and/or *_gnf.json from a previous suite",
    )
    parser.add_argument("--out-dir", required=True, help="Directory for rescored confusion outputs")
    parser.add_argument("--topk", type=int, default=3, help="Top-k predictions to store")
    args = parser.parse_args()

    source_dir = Path(args.source_suite_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = load_cases(source_dir)
    manifest = []
    for name, gnf_path in cases:
        gnf_payload = json.loads(gnf_path.read_text(encoding="utf-8"))
        if "d" not in gnf_payload or "gnf_factors" not in gnf_payload:
            raise ValueError(f"{gnf_path} must contain 'd' and 'gnf_factors'")
        gnf_payload = dict(gnf_payload)
        gnf_payload["source_gnf_path"] = str(gnf_path)
        gnf_payload["num_factors"] = len(gnf_payload["gnf_factors"])
        emit_case(
            checkpoint=args.checkpoint,
            device=args.device,
            out_dir=out_dir,
            name=name,
            gnf_payload=gnf_payload,
            topk=args.topk,
        )
        manifest.append({"name": name, "gnf_path": str(out_dir / f"{name}_gnf.json")})

    write_json(out_dir / "manifest.json", {"cases": manifest, "source_suite_dir": str(source_dir)})
    print(json.dumps({"out_dir": str(out_dir), "num_cases": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
