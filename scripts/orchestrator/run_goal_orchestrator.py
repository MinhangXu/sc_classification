#!/usr/bin/env python
"""
Stage-based orchestrator for long-running comprehensive studies.

Features:
- reads a goal spec (yaml/json)
- executes stage commands with retries
- supports validation gates (file/glob existence)
- writes RUN_STATE.json after each stage
- supports resume from previous state
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


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
    working_dir: Optional[str]
    timeout_seconds: Optional[int]
    retries: int
    validations: List[Dict[str, Any]]


def _extract_stages(spec: Dict[str, Any]) -> List[Stage]:
    stages_raw = spec.get("stages", [])
    default_working_dir = spec.get("working_dir")
    default_timeout = spec.get("timeout_seconds")
    default_retries = int(spec.get("retries", 0))
    stages: List[Stage] = []
    for item in stages_raw:
        name = str(item.get("name", "")).strip()
        cmd = str(item.get("command", "")).strip()
        if not name or not cmd:
            continue
        stages.append(
            Stage(
                name=name,
                command=cmd,
                working_dir=item.get("working_dir", default_working_dir),
                timeout_seconds=item.get("timeout_seconds", default_timeout),
                retries=int(item.get("retries", default_retries)),
                validations=list(item.get("validations", [])),
            )
        )
    return stages


def _stage_log_path(state_path: Path, stage_name: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stage_name)
    return state_path.parent / "logs" / "stages" / f"{safe}.log"


def _resolve_stage_cwd(stage: Stage, spec_path: Path) -> Path:
    base = spec_path.parent
    if not stage.working_dir:
        return base
    p = Path(stage.working_dir)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _run_command(stage: Stage, stage_cwd: Path, timeout_seconds: Optional[int], stage_log: Path) -> Dict[str, Any]:
    stage_log.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        stage.command,
        cwd=stage_cwd.as_posix(),
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    with stage_log.open("a") as f:
        f.write(f"[{_utc_now()}] command: {stage.command}\n")
        f.write(f"[{_utc_now()}] cwd: {stage_cwd}\n")
        f.write(f"[{_utc_now()}] exit_code: {proc.returncode}\n")
        if proc.stdout:
            f.write("\n--- stdout ---\n")
            f.write(proc.stdout)
            if not proc.stdout.endswith("\n"):
                f.write("\n")
        if proc.stderr:
            f.write("\n--- stderr ---\n")
            f.write(proc.stderr)
            if not proc.stderr.endswith("\n"):
                f.write("\n")
    return {"exit_code": int(proc.returncode)}


def _run_validations(stage: Stage, stage_cwd: Path, validation_log: Path) -> Dict[str, Any]:
    checks = stage.validations or []
    if not checks:
        _append_log(validation_log, f"{stage.name}: no validations configured")
        return {"passed": True, "checks": []}

    details: List[Dict[str, Any]] = []
    all_passed = True
    for check in checks:
        ctype = str(check.get("type", "")).strip()
        result = {"type": ctype, "ok": False}

        if ctype == "file_exists":
            raw_path = str(check.get("path", "")).strip()
            target = Path(raw_path)
            if not target.is_absolute():
                target = (stage_cwd / target).resolve()
            ok = target.exists() and target.is_file()
            result.update({"path": target.as_posix(), "ok": ok})

        elif ctype == "glob_exists":
            pattern = str(check.get("pattern", "")).strip()
            root_raw = str(check.get("root", ".")).strip()
            root = Path(root_raw)
            if not root.is_absolute():
                root = (stage_cwd / root).resolve()
            matches = [p.as_posix() for p in root.glob(pattern)]
            ok = len(matches) > 0
            result.update({"root": root.as_posix(), "pattern": pattern, "count": len(matches), "ok": ok})

        else:
            result.update({"ok": False, "error": f"unsupported validation type: {ctype}"})

        all_passed = all_passed and bool(result.get("ok"))
        details.append(result)
        _append_log(validation_log, f"{stage.name}: {json.dumps(result)}")

    return {"passed": all_passed, "checks": details}


def _load_state_if_exists(state_path: Path) -> Optional[Dict[str, Any]]:
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text())


def _find_stage_start_index(stages: List[Stage], state: Dict[str, Any]) -> int:
    stage_results = state.get("stage_results", {})
    # Resume from first stage that is not completed.
    for idx, stage in enumerate(stages):
        rec = stage_results.get(stage.name, {})
        if rec.get("status") != "completed":
            return idx
    return len(stages)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-based run orchestrator")
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
    parser.add_argument(
        "--resume-from-state",
        action="store_true",
        help="Resume from existing state file instead of starting a fresh run",
    )
    args = parser.parse_args()

    spec_path = Path(args.spec).resolve()
    state_path = Path(args.state).resolve()
    log_path = state_path.parent / "logs" / "agent_run.log"

    spec = _read_spec(spec_path)
    stages = _extract_stages(spec)
    if not stages:
        raise SystemExit("No valid stages found in spec.")

    state = _load_state_if_exists(state_path) if args.resume_from_state else None
    if state is None:
        state = {
            "spec": str(spec_path),
            "objective": spec.get("objective", ""),
            "status": "running",
            "started_at_utc": _utc_now(),
            "last_completed_stage": None,
            "next_stage": stages[0].name,
            "dry_run": bool(args.dry_run),
            "stage_results": {},
        }
        _append_log(log_path, f"orchestrator started: spec={spec_path} dry_run={args.dry_run}")
        start_idx = 0
    else:
        if str(state.get("spec", "")) != str(spec_path):
            raise SystemExit("State/spec mismatch. Refusing to resume with a different spec path.")
        if state.get("status") == "completed":
            _append_log(log_path, "resume requested but run is already completed")
            print("Run already completed. Nothing to do.")
            return
        state["status"] = "running"
        state["dry_run"] = bool(args.dry_run)
        _append_log(log_path, f"orchestrator resumed: spec={spec_path} dry_run={args.dry_run}")
        start_idx = _find_stage_start_index(stages, state)

    _write_json(state_path, state)
    validation_log = state_path.parent / "logs" / "validation.log"
    decision_log = state_path.parent / "logs" / "decision.log"

    stop_on_stage_failure = bool(spec.get("stop_on_stage_failure", True))
    stop_on_validation_failure = bool(spec.get("stop_on_validation_failure", True))

    for i in range(start_idx, len(stages)):
        stage = stages[i]
        stage_cwd = _resolve_stage_cwd(stage, spec_path)
        stage_log = _stage_log_path(state_path, stage.name)
        _append_log(log_path, f"stage start: {stage.name} :: {stage.command}")
        _append_log(log_path, f"stage cwd: {stage_cwd}")

        attempts = 0
        max_attempts = max(stage.retries, 0) + 1
        exit_code = 0
        run_error = None

        while attempts < max_attempts:
            attempts += 1
            _append_log(log_path, f"stage attempt {attempts}/{max_attempts}: {stage.name}")
            if args.dry_run:
                stage_log.parent.mkdir(parents=True, exist_ok=True)
                with stage_log.open("a") as f:
                    f.write(f"[{_utc_now()}] dry_run command: {stage.command}\n")
                exit_code = 0
            else:
                try:
                    run_info = _run_command(stage, stage_cwd=stage_cwd, timeout_seconds=stage.timeout_seconds, stage_log=stage_log)
                    exit_code = int(run_info["exit_code"])
                except subprocess.TimeoutExpired:
                    exit_code = 124
                    run_error = f"timeout after {stage.timeout_seconds}s"
                    _append_log(log_path, f"stage timeout: {stage.name} ({run_error})")
                except Exception as exc:
                    exit_code = 1
                    run_error = f"exception: {exc}"
                    _append_log(log_path, f"stage exception: {stage.name} ({run_error})")

            if exit_code == 0:
                break

        stage_result = {
            "attempts": attempts,
            "max_attempts": max_attempts,
            "exit_code": exit_code,
            "run_error": run_error,
            "completed_at_utc": _utc_now(),
        }
        state["stage_results"][stage.name] = stage_result

        if exit_code != 0:
            stage_result["status"] = "failed"
            state["status"] = "blocked"
            state["next_stage"] = stage.name
            _append_log(decision_log, f"{stage.name}: failed with exit_code={exit_code}")
            _write_json(state_path, state)
            if stop_on_stage_failure:
                _append_log(log_path, f"orchestrator stopped on stage failure: {stage.name}")
                print(f"Stage failed: {stage.name}. See logs.")
                return
            continue

        validation_result = _run_validations(stage, stage_cwd=stage_cwd, validation_log=validation_log)
        stage_result["validation"] = validation_result

        if not validation_result["passed"]:
            stage_result["status"] = "validation_failed"
            state["status"] = "blocked"
            state["next_stage"] = stage.name
            _append_log(decision_log, f"{stage.name}: validation failed")
            _write_json(state_path, state)
            if stop_on_validation_failure:
                _append_log(log_path, f"orchestrator stopped on validation failure: {stage.name}")
                print(f"Validation failed: {stage.name}. See logs.")
                return
            continue

        stage_result["status"] = "completed"
        state["last_completed_stage"] = stage.name
        state["next_stage"] = stages[i + 1].name if i + 1 < len(stages) else None
        _append_log(decision_log, f"{stage.name}: completed")
        _write_json(state_path, state)

    state["status"] = "completed"
    state["finished_at_utc"] = _utc_now()
    _write_json(state_path, state)
    _append_log(log_path, "orchestrator completed")
    print("Orchestrator completed.")


if __name__ == "__main__":
    main()

