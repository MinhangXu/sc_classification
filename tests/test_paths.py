"""Tests for src/sc_classification/utils/paths.py (stdlib-only path resolution).

Every test except the last builds a throwaway repo in tmp_path and monkeypatches the SC_*
variables, so nothing depends on the lab server. The last test re-checks the real canonical
experiment read-only and is skipped when that directory is absent.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PATHS_FILE = REPO / "src" / "sc_classification" / "utils" / "paths.py"


def _load_paths():
    # Load by file path, the same way tools/research_index.py does, so the test does not
    # depend on the package being installed.
    spec = importlib.util.spec_from_file_location("sc_paths_under_test", PATHS_FILE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


paths = _load_paths()
CANON = paths.CANONICAL_EXPERIMENT_ID
FALLBACK = paths._FALLBACK_CONFIGS[paths.DEFAULT_STUDY]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in paths.ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _make_repo(base: Path, name: str = "sc_classification") -> Path:
    repo = base / "clone"
    study = repo / "scripts" / "mrd_stage0_2"
    for sub in ("stage2_supervised", "stage0_panels", "workstreams/W3_dr_value_regularization"):
        (study / sub).mkdir(parents=True)
    (repo / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools"]\n\n[project]\nname = "{}"\nversion = "0"\n'.format(name)
    )
    return repo


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    monkeypatch.setenv("SC_REPO_ROOT", str(repo))
    return repo


def _write_config(repo: Path, **values) -> Path:
    path = repo / "scripts" / "mrd_stage0_2" / "research" / "study_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values))
    return path


# --------------------------------------------------------------------------------------
# repo_root
# --------------------------------------------------------------------------------------


def test_repo_root_default_is_this_checkout():
    assert paths.repo_root() == REPO


def test_repo_root_env_override(fake_repo):
    assert paths.repo_root() == fake_repo
    assert paths.study_dir() == fake_repo / "scripts" / "mrd_stage0_2"


def test_repo_root_env_missing_dir_names_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("SC_REPO_ROOT", str(tmp_path / "absent"))
    with pytest.raises(FileNotFoundError, match="SC_REPO_ROOT"):
        paths.repo_root()


def test_repo_root_walks_up_past_other_projects(tmp_path):
    repo = _make_repo(tmp_path)
    nested = repo / "scripts" / "mrd_stage0_2" / "stage0_panels" / "vendored"
    nested.mkdir()
    (nested / "pyproject.toml").write_text('[project]\nname = "something_else"\n')
    assert paths.repo_root(start=nested) == repo
    assert paths.repo_root(start=nested / "pyproject.toml") == repo


def test_repo_root_requires_project_name(tmp_path):
    other = _make_repo(tmp_path, name="not_sc_classification")
    with pytest.raises(FileNotFoundError, match="SC_REPO_ROOT"):
        paths.repo_root(start=other / "scripts")


# --------------------------------------------------------------------------------------
# defaults from repo_root and study_config
# --------------------------------------------------------------------------------------


def test_defaults_without_study_config(fake_repo):
    assert not paths.study_config_path().exists()
    assert paths.study_config()["canonical_experiment_id"] == CANON
    assert paths.experiment_root() == fake_repo / "experiments"
    assert paths.experiment_id() == CANON
    exp = fake_repo / "experiments" / CANON
    assert paths.experiment_dir(must_exist=False) == exp
    assert paths.data_root() == fake_repo.parent / "data"
    assert paths.cohort_h5ad() == fake_repo.parent / "data" / FALLBACK["raw_h5ad_relpath"]
    assert paths.cohort_cache() == exp / FALLBACK["cohort_cache_relpath"]
    assert paths.cell_metadata() == exp / FALLBACK["cell_metadata_relpath"]


def test_defaults_from_study_config(fake_repo):
    _write_config(
        fake_repo,
        study="mrd_stage0_2",
        canonical_experiment_id="20990101_000000_fixture_deadbeef",
        raw_h5ad_relpath="cohort/raw.h5ad",
        cohort_cache_relpath="analysis/cache/x/cohort.h5ad",
    )
    exp = fake_repo / "experiments" / "20990101_000000_fixture_deadbeef"
    assert paths.study_config()["canonical_experiment_id"] == "20990101_000000_fixture_deadbeef"
    assert paths.experiment_dir(must_exist=False) == exp
    assert paths.cohort_h5ad() == fake_repo.parent / "data" / "cohort" / "raw.h5ad"
    assert paths.cohort_cache() == exp / "analysis" / "cache" / "x" / "cohort.h5ad"
    # keys missing from the config fall back to the built-in value
    assert paths.cell_metadata() == exp / FALLBACK["cell_metadata_relpath"]


def test_study_config_returns_fresh_dict(fake_repo):
    _write_config(fake_repo, canonical_experiment_id="E1")
    paths.study_config()["canonical_experiment_id"] = "mutated"
    assert paths.experiment_id() == "E1"
    paths.study_config()["canonical_experiment_id"] = "mutated"  # fallback copy too
    assert FALLBACK["canonical_experiment_id"] == CANON


def test_malformed_study_config_raises(fake_repo):
    path = _write_config(fake_repo)
    path.write_text("{not json")
    with pytest.raises(ValueError, match="study_config.json"):
        paths.study_config()


def test_unknown_study_needs_experiment_id(fake_repo, monkeypatch):
    with pytest.raises(KeyError, match="SC_EXPERIMENT_ID"):
        paths.experiment_id(study="other_study")
    monkeypatch.setenv("SC_EXPERIMENT_ID", "E2")
    assert paths.experiment_dir(study="other_study", must_exist=False) == fake_repo / "experiments" / "E2"


# --------------------------------------------------------------------------------------
# environment variable overrides
# --------------------------------------------------------------------------------------


def test_env_overrides_each_variable(fake_repo, tmp_path, monkeypatch):
    exp_root = tmp_path / "elsewhere" / "experiments"
    data = tmp_path / "elsewhere" / "data"
    h5ad = tmp_path / "raw" / "cohort.h5ad"
    monkeypatch.setenv("SC_EXPERIMENT_ROOT", str(exp_root))
    assert paths.experiment_root() == exp_root
    assert paths.experiment_dir(must_exist=False) == exp_root / CANON
    monkeypatch.setenv("SC_EXPERIMENT_ID", "E3")
    assert paths.experiment_dir(must_exist=False) == exp_root / "E3"
    assert paths.cohort_cache() == exp_root / "E3" / FALLBACK["cohort_cache_relpath"]
    monkeypatch.setenv("SC_DATA_ROOT", str(data))
    assert paths.data_root() == data
    assert paths.cohort_h5ad() == data / FALLBACK["raw_h5ad_relpath"]
    monkeypatch.setenv("SC_COHORT_H5AD", str(h5ad))
    assert paths.cohort_h5ad() == h5ad


def test_env_empty_means_default(fake_repo, monkeypatch):
    for var in ("SC_EXPERIMENT_ROOT", "SC_EXPERIMENT_ID", "SC_DATA_ROOT", "SC_COHORT_H5AD"):
        monkeypatch.setenv(var, "  ")
    assert paths.experiment_dir(must_exist=False) == fake_repo / "experiments" / CANON
    assert paths.data_root() == fake_repo.parent / "data"


def test_env_values_expand_user_vars_and_relative(fake_repo, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SC_EXPERIMENT_ROOT", os.path.join("~", "exps"))
    assert paths.experiment_root() == tmp_path / "home" / "exps"
    monkeypatch.setenv("SC_FIXTURE_BASE", str(tmp_path / "base"))
    monkeypatch.setenv("SC_DATA_ROOT", os.path.join("$SC_FIXTURE_BASE", "data"))
    assert paths.data_root() == tmp_path / "base" / "data"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SC_DATA_ROOT", "rel_data")
    assert paths.data_root() == tmp_path / "rel_data"


# --------------------------------------------------------------------------------------
# experiment_dir errors
# --------------------------------------------------------------------------------------


def test_experiment_dir_missing_names_variables(fake_repo):
    with pytest.raises(FileNotFoundError) as excinfo:
        paths.experiment_dir()
    message = str(excinfo.value)
    assert "SC_EXPERIMENT_ROOT" in message and "SC_EXPERIMENT_ID" in message
    assert CANON in message


def test_experiment_dir_exists_and_explicit_id(fake_repo):
    exp = fake_repo / "experiments" / CANON
    exp.mkdir(parents=True)
    assert paths.experiment_dir() == exp
    other = fake_repo / "experiments" / "E4"
    other.mkdir()
    assert paths.experiment_dir("E4") == other
    with pytest.raises(FileNotFoundError, match="SC_EXPERIMENT_ID"):
        paths.experiment_dir("E5")
    with pytest.raises(FileNotFoundError, match="cohort_cache_relpath"):
        paths.cohort_cache(must_exist=True)


# --------------------------------------------------------------------------------------
# rel_to_experiment
# --------------------------------------------------------------------------------------


def test_rel_to_experiment_round_trip(fake_repo, monkeypatch):
    exp = fake_repo / "experiments" / CANON
    target = exp / "analysis" / "scorecards" / "stage2_diagnostic" / "slug_v1" / "table.csv"
    rel = paths.rel_to_experiment(target)
    assert rel == "analysis/scorecards/stage2_diagnostic/slug_v1/table.csv"
    assert paths.experiment_dir(must_exist=False) / rel == target
    assert paths.rel_to_experiment(str(target)) == rel
    assert paths.rel_to_experiment(exp) == "."
    # relative inputs are taken relative to the working directory
    exp.mkdir(parents=True)
    monkeypatch.chdir(exp)
    assert paths.rel_to_experiment(Path("analysis") / "x.csv") == "analysis/x.csv"
    # explicit exp_id
    assert paths.rel_to_experiment(fake_repo / "experiments" / "E6" / "a.txt", exp_id="E6") == "a.txt"


def test_rel_to_experiment_rejects_outside(fake_repo):
    exp = fake_repo / "experiments" / CANON
    with pytest.raises(ValueError, match="outside the experiment dir"):
        paths.rel_to_experiment(fake_repo / "experiments" / "other" / "x.csv")
    with pytest.raises(ValueError):
        paths.rel_to_experiment(exp / ".." / "other" / "x.csv")
    with pytest.raises(ValueError):
        paths.rel_to_experiment(fake_repo / "experiments" / (CANON + "_v2") / "x.csv")


def test_rel_to_experiment_follows_symlinks(fake_repo, tmp_path):
    exp = fake_repo / "experiments" / CANON
    (exp / "analysis").mkdir(parents=True)
    link = tmp_path / "exp_link"
    try:
        link.symlink_to(exp, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    assert paths.rel_to_experiment(link / "analysis") == "analysis"


# --------------------------------------------------------------------------------------
# add_study_module_paths
# --------------------------------------------------------------------------------------


def test_add_study_module_paths_idempotent(fake_repo, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    study = fake_repo / "scripts" / "mrd_stage0_2"
    expected = [
        str(study / "stage2_supervised"),
        str(study / "stage0_panels"),
        str(study / "workstreams" / "W3_dr_value_regularization"),
    ]
    before = list(sys.path)
    paths.add_study_module_paths(workstreams=("W3_dr_value_regularization",))
    assert sys.path[:3] == expected
    assert sys.path[3:] == before
    snapshot = list(sys.path)
    paths.add_study_module_paths(workstreams=("W3_dr_value_regularization",))
    paths.add_study_module_paths()
    paths.add_study_module_paths(workstreams=("W3",))  # bare id resolves to the same dir
    assert sys.path == snapshot
    for entry in expected:
        assert sys.path.count(entry) == 1


def test_add_study_module_paths_missing_dir_raises(fake_repo, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    before = list(sys.path)
    with pytest.raises(FileNotFoundError, match="W9"):
        paths.add_study_module_paths(workstreams=("W9",))
    with pytest.raises(FileNotFoundError, match="nope"):
        paths.add_study_module_paths(subdirs=("nope",))
    assert sys.path == before


# --------------------------------------------------------------------------------------
# git_state
# --------------------------------------------------------------------------------------


def test_git_state_outside_work_tree(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    plain = tmp_path / "plain"
    plain.mkdir()
    assert paths.git_state(plain) == (None, None)


@pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")
def test_git_state_head_and_dirty(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    repo = tmp_path / "gitrepo"
    repo.mkdir()

    def git(*args):
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-C", str(repo), *args],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    git("init", "-q")
    (repo / "a.txt").write_text("a\n")
    git("add", "a.txt")
    git("commit", "-q", "-m", "init")
    head, dirty = paths.git_state(repo)
    assert head is not None and 7 <= len(head) <= 40
    int(head, 16)
    assert dirty is False
    (repo / "b.txt").write_text("b\n")
    assert paths.git_state(repo) == (head, True)


# --------------------------------------------------------------------------------------
# diagnostics, compatibility, real experiment
# --------------------------------------------------------------------------------------


def test_main_prints_every_path(fake_repo, capsys):
    assert paths.main([]) == 0
    out = capsys.readouterr().out
    for name in (
        "repo_root",
        "study_config",
        "experiment_root",
        "experiment_dir",
        "data_root",
        "cohort_h5ad",
        "cohort_cache",
        "cell_metadata",
        "import pytest",
        "import dash",
    ):
        assert name in out
    assert "exists=False" in out and "exists=True" in out
    for var in paths.ENV_VARS:
        assert var in out
    rows = paths.diagnostics()
    assert {row["name"] for row in rows} >= {"repo_root", "experiment_dir", "cohort_cache"}


def test_diagnostics_never_raise(tmp_path, monkeypatch):
    monkeypatch.setenv("SC_REPO_ROOT", str(tmp_path / "absent"))
    rows = paths.diagnostics()
    errors = {row["name"]: row["error"] for row in rows}
    assert errors["repo_root"] is not None and "SC_REPO_ROOT" in errors["repo_root"]
    assert paths.main([]) == 0


def test_source_parses_as_python38():
    ast.parse(PATHS_FILE.read_text(encoding="utf-8"), feature_version=(3, 8))


def _real_experiment() -> Path:
    config = REPO / "scripts" / "mrd_stage0_2" / "research" / "study_config.json"
    exp_id = CANON
    if config.is_file():
        exp_id = json.loads(config.read_text()).get("canonical_experiment_id", CANON)
    return REPO / "experiments" / exp_id


@pytest.mark.skipif(not _real_experiment().is_dir(), reason="canonical experiment dir not present")
def test_real_experiment_resolves_read_only():
    exp = paths.experiment_dir()
    assert exp == _real_experiment()
    rel = paths.rel_to_experiment(paths.cohort_cache())
    assert not rel.startswith(("/", "..", "experiments/"))
    assert paths.cohort_cache() == exp / rel
    assert paths.cell_metadata(must_exist=True).is_file()
