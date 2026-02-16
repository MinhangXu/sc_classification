#!/usr/bin/env python
"""
Minimal stage-based orchestrator scaffold for comprehensive runs.

This script intentionally starts simple:
- reads a goal spec (yaml/json)
- logs stage transitions
- writes RUN_STATE.json
- supports resume from state file

You can later wire each stage to concrete commands such as:
  scripts/comprehensive_run/run_gene_filter_dr_grid.py plan0
  scripts/comprehensive_run/run_gene_filter_dr_grid.py plan1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_spec(path: Path) -> Dict[str, Any]:
    text = path.read_text()
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # optional dependency
    except Exception as exc:
        raise RuntimeError(
            "YAML spec requested but PyYAML is not available. "
            "Use JSON spec or install pyyaml."
        ) from exc
    return yaml.safe_load(text)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(f"[{_utc_now()}] {message}\n")


@dataclass
class Stage:
    name: str
    command: str


def _extract_stages(spec: Dict[str, Any]) -> List[Stage]:
    stages_raw = spec.get("stages", [])
    stages: List[Stage] = []
    for item in stages_raw:
        name = str(item.get("name", "")).strip()
        cmd = str(item.get("command", "")).strip()
        if not name or not cmd:
            continue
        stages.append(Stage(name=name, command=cmd))
    return stages


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-based run orchestrator scaffold")
    parser.add_argument("--spec", required=True, help="Path to run spec (.yaml or .json)")
    parser.add_argument(
        "--state",
        default="RUN_STATE.json",
        help="Path to state file (default: RUN_STATE.json in cwd)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not execute commands; only log planned execution",
    )
    args = parser.parse_args()

    spec_path = Path(args.spec).resolve()
    state_path = Path(args.state).resolve()
    log_path = state_path.parent / "logs" / "agent_run.log"

    spec = _read_spec(spec_path)
    stages = _extract_stages(spec)
    if not stages:
        raise SystemExit("No valid stages found in spec.")

    state: Dict[str, Any] = {
        "spec": str(spec_path),
        "status": "running",
        "started_at_utc": _utc_now(),
        "last_completed_stage": None,
        "next_stage": stages[0].name,
        "dry_run": bool(args.dry_run),
    }
    _write_json(state_path, state)
    _append_log(log_path, f"orchestrator started: spec={spec_path} dry_run={args.dry_run}")

    for i, stage in enumerate(stages):
        _append_log(log_path, f"stage start: {stage.name} :: {stage.command}")
        # Scaffold behavior only: actual command execution is intentionally deferred.
        if args.dry_run:
            _append_log(log_path, f"stage dry-run complete: {stage.name}")
        else:
            _append_log(
                log_path,
                f"stage skipped execution (scaffold): {stage.name}; wire subprocess in next increment",
            )

        state["last_completed_stage"] = stage.name
        state["next_stage"] = stages[i + 1].name if i + 1 < len(stages) else None
        _write_json(state_path, state)

    state["status"] = "completed"
    state["finished_at_utc"] = _utc_now()
    _write_json(state_path, state)
    _append_log(log_path, "orchestrator completed")


if __name__ == "__main__":
    main()

