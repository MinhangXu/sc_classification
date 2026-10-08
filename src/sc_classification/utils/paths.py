"""Portable path resolution for NEW sc_classification code (stdlib only).

New workstream scripts, notebooks, apps and ``tools/research_index.py`` resolve every
location through this module instead of hard-coding paths. Older code is not rewritten;
see AGENTS.md (legacy path debt) and ``python tools/research_index.py portability``.

Never add absolute or home-relative paths to this file (research_index lint E07).

Environment variables (all optional; documented in ``.env.example``). An unset or empty
variable means "use the default". Values may use ``~`` and ``$VARS``; relative values are
taken relative to the current working directory.

    SC_REPO_ROOT        repo root. Default: walk up from this file (then from the current
                        working directory) to the directory whose pyproject.toml declares
                        name = "sc_classification".
    SC_EXPERIMENT_ROOT  directory that holds experiment dirs. Default: <repo>/experiments.
    SC_EXPERIMENT_ID    experiment dir name. Default: canonical_experiment_id from
                        scripts/<study>/research/study_config.json, falling back to
                        CANONICAL_EXPERIMENT_ID for the default study.
    SC_DATA_ROOT        umbrella data dir. Default: <repo>/../data.
    SC_COHORT_H5AD      raw cohort AnnData. Default: SC_DATA_ROOT/<raw_h5ad_relpath>.

Study constants (canonical id, supported patients, EXP-relative cache paths, protected
paths, ...) live in scripts/<study>/research/study_config.json. When that file does not
exist yet, the built-in fallbacks below are used for the default study.

Diagnostics: ``python -m sc_classification.utils.paths [STUDY]`` prints every resolved
path with an exists flag, the git state, and whether pytest and dash import.

The module must stay importable on Python 3.8 because research_index.py loads it by file
path with importlib.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

PathLike = Union[str, "os.PathLike[str]"]

PROJECT_NAME = "sc_classification"
DEFAULT_STUDY = "mrd_stage0_2"
CANONICAL_EXPERIMENT_ID = "20260525_060508_stage0_mrd_old34_broad_screen_82db5093"

ENV_REPO_ROOT = "SC_REPO_ROOT"
ENV_EXPERIMENT_ROOT = "SC_EXPERIMENT_ROOT"
ENV_EXPERIMENT_ID = "SC_EXPERIMENT_ID"
ENV_DATA_ROOT = "SC_DATA_ROOT"
ENV_COHORT_H5AD = "SC_COHORT_H5AD"
ENV_VARS = (ENV_REPO_ROOT, ENV_EXPERIMENT_ROOT, ENV_EXPERIMENT_ID, ENV_DATA_ROOT, ENV_COHORT_H5AD)

# Fallbacks used only when study_config.json is absent or lacks the key. They mirror the
# values recorded in scripts/mrd_stage0_2/research/study_config.json.
_FALLBACK_CONFIGS: Dict[str, Dict[str, Any]] = {
    DEFAULT_STUDY: {
        "study": DEFAULT_STUDY,
        "canonical_experiment_id": CANONICAL_EXPERIMENT_ID,
        "raw_h5ad_relpath": "cohort_adata/adata_cellType_cnLabel_pseudoTime_collectionTime.h5ad",
        "cohort_cache_relpath": (
            "analysis/cache/stage2_diagnostic/deg_hvg_lowsupport_v1/mrd_cite_binary_log1p_15560.h5ad"
        ),
        "cell_metadata_relpath": (
            "stage2_supervised/multiobjective/discovery_full_cohort/cell_metadata.csv.gz"
        ),
    }
}

DEFAULT_STUDY_SUBDIRS = ("stage2_supervised", "stage0_panels")


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _env(name: str) -> Optional[str]:
    """Return the stripped value of an env var, or None when unset or empty."""
    value = os.environ.get(name, "").strip()
    return value or None


def _env_path(name: str) -> Optional[Path]:
    value = _env(name)
    if value is None:
        return None
    expanded = os.path.expandvars(os.path.expanduser(value))
    return Path(os.path.abspath(expanded))


def _is_project_root(candidate: Path) -> bool:
    """True when candidate/pyproject.toml declares [project] name = "sc_classification"."""
    pyproject = candidate / "pyproject.toml"
    if not pyproject.is_file():
        return False
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return False
    section = ""
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]").strip()
            continue
        if section != "project" or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key == "name":
            return value.strip("\"'") == PROJECT_NAME
    return False


def _walk_up(start: Path) -> Optional[Path]:
    start = Path(os.path.abspath(str(start)))
    if start.is_file():
        start = start.parent
    for candidate in (start, *start.parents):
        if _is_project_root(candidate):
            return candidate
    return None


# --------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------


def repo_root(start: Optional[PathLike] = None) -> Path:
    """Return the sc_classification repo root.

    Order: SC_REPO_ROOT if set; else walk up from ``start`` when given; else walk up from
    this file and then from the current working directory, stopping at the first directory
    whose pyproject.toml declares name = "sc_classification".
    """
    env_root = _env_path(ENV_REPO_ROOT)
    if env_root is not None:
        if not env_root.is_dir():
            raise FileNotFoundError(
                "{} points to a missing directory: {}. Unset it to auto-detect the repo root, "
                "or set it to your sc_classification clone.".format(ENV_REPO_ROOT, env_root)
            )
        return env_root
    starts: List[Path] = [Path(start)] if start is not None else [Path(__file__), Path.cwd()]
    for origin in starts:
        found = _walk_up(origin)
        if found is not None:
            return found
    raise FileNotFoundError(
        "Could not find a pyproject.toml with name = \"{}\" above {}. Set {} to your "
        "sc_classification clone.".format(PROJECT_NAME, ", ".join(str(s) for s in starts), ENV_REPO_ROOT)
    )


def study_dir(study: str = DEFAULT_STUDY) -> Path:
    """Return scripts/<study> under the repo root (not required to exist)."""
    return repo_root() / "scripts" / study


def study_config_path(study: str = DEFAULT_STUDY) -> Path:
    """Return scripts/<study>/research/study_config.json (not required to exist)."""
    return study_dir(study) / "research" / "study_config.json"


def study_config(study: str = DEFAULT_STUDY) -> Dict[str, Any]:
    """Return the study constants.

    Reads scripts/<study>/research/study_config.json when it exists (a fresh dict on every
    call). Otherwise returns a copy of the built-in fallback for that study, which for the
    default study holds the canonical experiment id and the cache relpaths. Unknown studies
    without a config return ``{"study": study}``.
    """
    path = study_config_path(study)
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ValueError("Malformed study config {}: {}".format(path, exc)) from exc
        if not isinstance(data, dict):
            raise ValueError("Study config {} must hold a JSON object".format(path))
        return data
    return dict(_FALLBACK_CONFIGS.get(study, {"study": study}))


def _config_value(study: str, key: str) -> Optional[Any]:
    value = study_config(study).get(key)
    if value in (None, ""):
        value = _FALLBACK_CONFIGS.get(study, {}).get(key)
    return value


def experiment_root() -> Path:
    """Return SC_EXPERIMENT_ROOT, or <repo>/experiments."""
    env_root = _env_path(ENV_EXPERIMENT_ROOT)
    return env_root if env_root is not None else repo_root() / "experiments"


def experiment_id(study: str = DEFAULT_STUDY) -> str:
    """Return SC_EXPERIMENT_ID, or the study's canonical_experiment_id."""
    env_id = _env(ENV_EXPERIMENT_ID)
    if env_id is not None:
        return env_id
    canonical = _config_value(study, "canonical_experiment_id")
    if not canonical:
        raise KeyError(
            "Study {!r} has no canonical_experiment_id in {}. Set {} or pass exp_id.".format(
                study, study_config_path(study), ENV_EXPERIMENT_ID
            )
        )
    return str(canonical)


def experiment_dir(
    exp_id: Optional[str] = None, study: str = DEFAULT_STUDY, must_exist: bool = True
) -> Path:
    """Return the experiment dir: experiment_root() / (exp_id or experiment_id(study)).

    Raises FileNotFoundError naming SC_EXPERIMENT_ROOT and SC_EXPERIMENT_ID when
    ``must_exist`` and the directory is missing.
    """
    eid = exp_id if exp_id else experiment_id(study)
    root = experiment_root()
    path = root / eid
    if must_exist and not path.is_dir():
        root_src = "env" if _env(ENV_EXPERIMENT_ROOT) else "default <repo>/experiments"
        if exp_id:
            id_src = "exp_id argument"
        elif _env(ENV_EXPERIMENT_ID):
            id_src = "env"
        else:
            id_src = "study_config canonical_experiment_id"
        raise FileNotFoundError(
            "Experiment directory not found: {path}\n"
            "  {root_var}={root} ({root_src})\n"
            "  {id_var}={eid} ({id_src})\n"
            "Set {root_var} to the directory that holds experiment dirs and/or {id_var} to the "
            "experiment dir name (see .env.example).".format(
                path=path,
                root_var=ENV_EXPERIMENT_ROOT,
                root=root,
                root_src=root_src,
                id_var=ENV_EXPERIMENT_ID,
                eid=eid,
                id_src=id_src,
            )
        )
    return path


def data_root() -> Path:
    """Return SC_DATA_ROOT, or <repo>/../data (the umbrella layout)."""
    env_root = _env_path(ENV_DATA_ROOT)
    return env_root if env_root is not None else repo_root().parent / "data"


def cohort_h5ad(study: str = DEFAULT_STUDY, must_exist: bool = False) -> Path:
    """Return SC_COHORT_H5AD, or data_root() / study_config raw_h5ad_relpath."""
    env_file = _env_path(ENV_COHORT_H5AD)
    if env_file is not None:
        path = env_file
    else:
        relpath = _config_value(study, "raw_h5ad_relpath")
        if not relpath:
            raise KeyError(
                "Study {!r} has no raw_h5ad_relpath; set {}.".format(study, ENV_COHORT_H5AD)
            )
        path = data_root() / str(relpath)
    if must_exist and not path.is_file():
        raise FileNotFoundError(
            "Cohort AnnData not found: {}. Set {} to the file, or {} to the directory that "
            "holds it (see .env.example).".format(path, ENV_COHORT_H5AD, ENV_DATA_ROOT)
        )
    return path


def _exp_file(key: str, exp_id: Optional[str], study: str, must_exist: bool) -> Path:
    relpath = _config_value(study, key)
    if not relpath:
        raise KeyError("Study {!r} has no {} in {}".format(study, key, study_config_path(study)))
    path = experiment_dir(exp_id, study=study, must_exist=must_exist) / str(relpath)
    if must_exist and not path.is_file():
        raise FileNotFoundError(
            "{} not found: {} (EXP-relative {}). Check {} and {}.".format(
                key, path, relpath, ENV_EXPERIMENT_ROOT, ENV_EXPERIMENT_ID
            )
        )
    return path


def cohort_cache(
    exp_id: Optional[str] = None, study: str = DEFAULT_STUDY, must_exist: bool = False
) -> Path:
    """Return EXP / study_config cohort_cache_relpath (the binary-label log1p cohort cache)."""
    return _exp_file("cohort_cache_relpath", exp_id, study, must_exist)


def cell_metadata(
    exp_id: Optional[str] = None, study: str = DEFAULT_STUDY, must_exist: bool = False
) -> Path:
    """Return EXP / study_config cell_metadata_relpath (per-cell labels and patients)."""
    return _exp_file("cell_metadata_relpath", exp_id, study, must_exist)


def _resolve_workstream_dir(base: Path, name: str) -> Path:
    candidate = base / name
    if candidate.is_dir():
        return candidate
    # Allow a bare id such as "W3" when exactly one W3_* directory exists.
    if "_" not in name and base.is_dir():
        matches = sorted(p for p in base.glob(name + "_*") if p.is_dir())
        if len(matches) == 1:
            return matches[0]
    raise FileNotFoundError("Workstream directory not found: {}".format(candidate))


def add_study_module_paths(
    study: str = DEFAULT_STUDY,
    subdirs: Iterable[str] = DEFAULT_STUDY_SUBDIRS,
    workstreams: Iterable[str] = (),
) -> None:
    """Put study module dirs at the front of sys.path, idempotently.

    Adds scripts/<study>/<subdir> for each subdir, then
    scripts/<study>/workstreams/<dir> for each workstream (full directory name such as
    "W3_dr_value_regularization", or a bare id such as "W3" when unambiguous). Entries
    already on sys.path are left where they are; new ones are inserted at the front in the
    order given. Raises FileNotFoundError for a missing directory.
    """
    if isinstance(subdirs, str):
        subdirs = (subdirs,)
    if isinstance(workstreams, str):
        workstreams = (workstreams,)
    base = study_dir(study)
    wanted: List[Path] = []
    for sub in subdirs:
        path = base / sub
        if not path.is_dir():
            raise FileNotFoundError("Study module directory not found: {}".format(path))
        wanted.append(path)
    for ws in workstreams:
        wanted.append(_resolve_workstream_dir(base / "workstreams", ws))
    new_entries: List[str] = []
    for path in wanted:
        entry = str(path)
        if entry not in sys.path and entry not in new_entries:
            new_entries.append(entry)
    sys.path[0:0] = new_entries


def rel_to_experiment(path: PathLike, exp_id: Optional[str] = None, study: str = DEFAULT_STUDY) -> str:
    """Return ``path`` relative to the experiment dir as a POSIX string.

    Relative inputs are taken relative to the current working directory. Symlinks are
    honoured on both sides. The experiment dir itself maps to ".". Raises ValueError when
    the path is outside the experiment dir. EXP need not exist.
    """
    exp = experiment_dir(exp_id, study=study, must_exist=False)
    target = Path(os.path.abspath(os.fspath(path)))
    exp_abs = Path(os.path.abspath(str(exp)))
    for left, right in ((target, exp_abs), (target.resolve(), exp_abs.resolve())):
        try:
            return left.relative_to(right).as_posix()
        except ValueError:
            continue
    raise ValueError(
        "Path is outside the experiment dir: {} (EXP = {}). Devlog and registry paths must be "
        "EXP-relative.".format(target, exp_abs)
    )


def git_state(repo: Optional[PathLike] = None) -> Tuple[Optional[str], Optional[bool]]:
    """Return (short HEAD, dirty) for ``repo`` (default: repo_root()).

    dirty is True when ``git status --porcelain`` prints anything. Either element is None
    when git is unavailable, the directory is not a work tree, or the call fails.
    """
    try:
        target = Path(repo) if repo is not None else repo_root()
    except FileNotFoundError:
        return None, None

    def _git(*args: str) -> Optional[str]:
        try:
            proc = subprocess.run(
                ["git", "-C", str(target), *args],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                universal_newlines=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return proc.stdout if proc.returncode == 0 else None

    head_out = _git("rev-parse", "--short", "HEAD")
    head = (head_out.strip() or None) if head_out is not None else None
    status_out = _git("status", "--porcelain")
    dirty = bool(status_out.strip()) if status_out is not None else None
    return head, dirty


# --------------------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------------------


def _probe(fn: Any) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return fn(), None
    except Exception as exc:  # diagnostics must never crash
        return None, "{}: {}".format(type(exc).__name__, exc)


def diagnostics(study: str = DEFAULT_STUDY) -> List[Dict[str, Any]]:
    """Return one row per resolved location: {name, value, exists, error}.

    ``value`` is a string (or None on error); ``exists`` is True/False for paths and None
    otherwise. Never raises; research_index.py doctor can reuse it.
    """
    rows: List[Dict[str, Any]] = []

    def add(name: str, fn: Any, is_path: bool = True) -> None:
        value, error = _probe(fn)
        exists = None
        if is_path and value is not None:
            exists = Path(value).exists()
        rows.append(
            {"name": name, "value": None if value is None else str(value), "exists": exists, "error": error}
        )

    for var in ENV_VARS:
        rows.append({"name": "env " + var, "value": _env(var) or "(unset)", "exists": None, "error": None})
    add("repo_root", repo_root)
    add("study_dir", lambda: study_dir(study))
    add("study_config", lambda: study_config_path(study))
    add("experiment_root", experiment_root)
    add("experiment_id", lambda: experiment_id(study), is_path=False)
    add("experiment_dir", lambda: experiment_dir(study=study, must_exist=False))
    add("data_root", data_root)
    add("cohort_h5ad", lambda: cohort_h5ad(study))
    add("cohort_cache", lambda: cohort_cache(study=study))
    add("cell_metadata", lambda: cell_metadata(study=study))
    for sub in DEFAULT_STUDY_SUBDIRS:
        add("module_dir " + sub, lambda sub=sub: study_dir(study) / sub)
    return rows


def _import_status(module: str) -> str:
    try:
        mod = __import__(module)
    except Exception as exc:  # report, never crash
        return "no ({})".format(type(exc).__name__)
    return "yes ({})".format(getattr(mod, "__version__", "version unknown"))


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if any(a in ("-h", "--help") for a in args):
        print("usage: python -m sc_classification.utils.paths [STUDY]")
        return 0
    study = args[0] if args else DEFAULT_STUDY
    print("sc_classification paths (study: {})".format(study))
    print("python: {}".format(sys.version.split()[0]))
    cfg, _ = _probe(lambda: study_config_path(study))
    source = "study_config.json" if cfg is not None and cfg.is_file() else "built-in fallback"
    print("study constants from: {}".format(source))
    rows = diagnostics(study)
    width = max(len(r["name"]) for r in rows + [{"name": "import pytest"}])
    for row in rows:
        if row["error"] is not None:
            shown = "ERROR " + row["error"]
        elif row["exists"] is None:
            shown = row["value"]
        else:
            shown = "{}  exists={}".format(row["value"], row["exists"])
        print("  {}  {}".format(row["name"].ljust(width), shown))
    head, dirty = git_state()
    print("  {}  head={} dirty={}".format("git".ljust(width), head, dirty))
    for module in ("pytest", "dash"):
        print("  {}  {}".format(("import " + module).ljust(width), _import_status(module)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
