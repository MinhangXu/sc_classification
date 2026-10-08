"""Tests for tools/research_index.py (the research-spine generator, validator and query tool).

Fixture tests build a minimal study ``scripts/demo`` in tmp_path and run the tool against it,
so they never depend on the real spine. One integration test runs ``check`` on the real repo
and is skipped when scripts/mrd_stage0_2/research/study_config.json is absent.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO / "tools" / "research_index.py"
TEMPLATE = REPO / "scripts" / "mrd_stage0_2" / "research" / "devlog" / "_TEMPLATE.md"
TODAY = "2026-10-08"
HOME_NEEDLE = "/" + "home" + "/"  # assembled so this overlay file passes lint E07


def load_tool():
    spec = importlib.util.spec_from_file_location("research_index", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ri = load_tool()

# ---------------------------------------------------------------------------------------
# fixture study
# ---------------------------------------------------------------------------------------

STUDY_CONFIG = """{
  "study": "demo",
  "canonical_experiment_id": "demo_exp",
  "cohort": {"n_cells": 15022, "n_patients": 2},
  "patients": {"P01": {"malignant": 45, "total": 11860}, "P02": {"malignant": 211, "total": 1462},
               "P05": {"malignant": 8, "total": 1700}},
  "supported_patients": ["P01", "P02"],
  "support_rule": {"min_malignant_cells": 10, "require_both_classes": true},
  "stage1_fit_scope": "transductive_all_eligible_cells",
  "protected_paths": ["analysis/scorecards/stage2_discovery_full_cohort_scorecard.csv"],
  "protected_since": "2026-06-05",
  "versioned_slug_roots": ["analysis/scorecards/stage2_diagnostic", "stage1_dr_variants"],
  "versioned_slug_rule_since": "2026-10-08"
}
"""

THEME = """---
study: demo
last_reviewed: 2026-10-08
canonical_experiment: demo_exp
---

# Demo theme

> No numbers here. Cite H-, D-, F- and devlog IDs; a quoted 0.5 inside a blockquote is fine.

## 0. Working thesis
Classification as a biological probe (H-S1-04, D14, W1.2, Option 1.3; devlog 2026-10-08_spine-adopted).
"""

HYPOTHESES = """# Demo hypotheses

Static fields only.

# Stage 1

## H-S1-04 — Supported patients share one Stage 1 basis.
- stage: s1
- questions: [Q1, Q2]
- analyses: [D14, D16]
- workstreams: [W1]
- alternatives: [ALT-TRANS, ALT-PREV]
- blocking: [W1.2]
- figures: [Fig3]
- gate_supports: descriptive overlap on the transductive basis
- gate_not: inductive deployment
- origin: canvas 2026-10-02b

# Stage 2

## H-S2-07 - PCA orthogonality pushes later components beyond one patient.
- stage: s2
- questions: [QI, Q3]
- analyses: [D1]
- workstreams: [W1]
- alternatives: []
- blocking: []
- figures: []
- gate_supports: none
- gate_not: do not cite as a mechanism
- origin: new 2026-10-08
"""

ALTERNATIVES = """alt_id,name,reviewer_question,standard_tests,typical_tasks
ALT-TRANS,held-out patient in the basis,Does transfer survive an inductive basis?,Inductive LOPO Stage 1,W1.2
ALT-PREV,prevalence/support,"Is the lift just prevalence, not biology?",Support-matched lift,W1.1
"""

ANALYSES_HEADER = ("id,family,title,question,questions,asked_by,asks,workstream,notebook,code,experiment_id,"
                   "artifact_dir,findings_md,figure_prefix,date,status,notes\n")
ANALYSES = ANALYSES_HEADER + (
    "D1,D,Atlas of lift,Which programs transfer?,Q1,not_recorded,,W1,scripts/demo/notebooks/atlas.ipynb,,demo_exp,"
    "analysis/scorecards/stage2_diagnostic,,D1,2026-07-24,historical,Root diagnostic tables\n"
    "D14,D,Per-patient factor-score overlap,Do factor scores overlap across patients?,Q1;Q2,user,A-20261008-01,W1,"
    "scripts/demo/notebooks/overlap.ipynb,scripts/demo/code/overlap.py,demo_exp,"
    "analysis/scorecards/stage2_diagnostic/patient_factor_overlap_v1,"
    "analysis/scorecards/stage2_diagnostic/patient_factor_overlap_v1/findings.md,D14,2026-10-02,current,Transductive basis\n"
    "D16,D,Factor x patient matrix,How is factor variance allocated to patients?,Q1,user,,W1,,,demo_exp,"
    "analysis/scorecards/stage2_diagnostic/factor_patient_matrix_v1,,D16,2026-10-08,planned,\n"
    "D1DD1,D1DD,Deep dive one,What drives the atlas?,Q1;Q3,not_recorded,,W1,scripts/demo/notebooks/atlas.ipynb,,demo_exp,"
    "analysis/scorecards/stage2_diagnostic/d1a_deep_dive,,D1DD1,2026-09-10,historical,\n"
    "F1,F,K trends,Does lift change with K?,Q3,meeting:2026-09-17,,W1,scripts/demo/notebooks/atlas.ipynb,,demo_exp,"
    "analysis/scorecards/stage2_diagnostic/feedback_deep_dive,,F1,2026-09-17,historical,\n"
)

CAMPAIGNS = (
    "namespace,set,experiment_id,scope,launched,finished,status,relaunch_policy,runner,resume_command,screen_session,"
    "code_commit_effective,recorded_commit,notes\n"
    "root,A,demo_exp,biological panels,2026-05-25,2026-06-05,complete,never,run_demo.py,none,,b60a6ee,b60a6ee,protected\n"
    "donor_v2,,demo_exp,donor sensitivity,2026-08-21,2026-09-03,complete,resume_only,run_donor.sh,"
    "\"bash run_donor.sh --experiment-dir \"\"$SC_EXPERIMENT_ROOT/$SC_EXPERIMENT_ID\"\" --resume\",,b60a6ee,43f19be,\n"
)

ASKS = """---
date: 2026-10-08
source: report
asked_by: user
capture: verbatim_from_report
original_location: team report TODO list
---

# Team report TODO list

- [A-20261008-01] "Test whether to aggregate patients." -> D14, W1.2, H-S1-04
- [A-20261008-02] (teammate:Ankit) "Is the gap -> just L1 at fixed C?" -> W1.1
- [A-20261008-03] "Write the theme up." -> W0.2, THEME
"""

WS0 = """---
id: W0
title: Context and provenance
status: active
questions: [meta]
---

# W0 — Context and provenance

## Backlog
- [W0.1] (P0) Adopt the spine — done when: check exits 0
- [W0.2] (P2) Write the theme — done when: THEME reviewed
"""

WS1 = """---
id: W1
title: Patient aggregation
status: active
questions: [Q1, Q2]
---

# W1 — Patient aggregation

## Charter
Should patients share one basis?

## Backlog
- [W1.1] (P0) D16 factor x patient matrix — done when: tables exist
  - detail sub-bullets are ignored
- [W1.2] (P0) D20 overlap v2 — done when: calls are written (depends: W1.1)
- [W1.3] (P1) Rerun on the new bases (depends: W1.2) (blocked: waiting for W2 bases)

## Code in this directory
none yet
"""

D14_ENTRY = """---
id: 2026-10-02_d14-overlap
date: 2026-10-02
type: analysis
title: D14 per-patient factor-score overlap test
asked_by: user
asks: [A-20261008-01]
implemented_by: unknown
questions: [Q1, Q2]
workstream: W1
analyses: [D14]
hypotheses: [H-S1-04]
verdicts: [H-S1-04=supported_descriptive]
alts: []
closes: []
advances: [W1.1]
claim_level: descriptive
compute: notebook-cpu
c_policy: not_applicable
training_n_changed: false
status: done
commit: b60a6ee  # retro: first commit containing the code
run_head: 43f19be
dirty_run: true
experiment: demo_exp
backfilled: true
date_source: mtime  # CSV mtimes
supersedes: []
---

## Question
> "Test whether to aggregate patients." (A-20261008-01)

## Why it matters
Q1 needs a reason to pool patients on one basis (H-S1-04).

## Estimand / null
Median pairwise histogram overlap per factor; permutation null within label.

## Implementation
- Code: scripts/demo/code/overlap.py, driven from `scripts/demo/notebooks/overlap.ipynb`.
- Ran on CPU in a notebook kernel.

## Artifacts
- analysis/scorecards/stage2_diagnostic/patient_factor_overlap_v1/, containing:
  - calls.csv

## Results
- [H-S1-04 supports] Calls: 45 aggregate_shared of 47 representations.
- [H-S1-04 qualifies] Biological overlap:
  - median 0.843
- IFN-alpha eta2 0.010 in normal cells.

## Claim gate
Supports (descriptive): pooling on one transductive basis is reasonable.
Does not support:
- inductive statements.

## Next steps
- W1.2: harden the test.
"""

ADOPT_ENTRY = """---
id: 2026-10-08_spine-adopted
date: 2026-10-08
type: decision
title: Adopt the research spine
asked_by: user
asks: [A-20261008-02]
implemented_by: claude-code
questions: [meta]
workstream: W0
analyses: []
hypotheses: []
verdicts: [H-S1-04=supported_descriptive, H-S2-07=retired]
alts: [H-S1-04:ALT-PREV=partial]
closes: []
advances: [W0.1]
claim_level: none
status: done
commit: pending
run_head: b60a6ee
dirty_run: true
experiment: demo_exp
backfilled: false
date_source: authored
supersedes: []
---

## Question
> "Adopt the spine." (A-20261008-02)

## Why it matters
All later entries cite it.

## Estimand / null
n/a

## Implementation
n/a

## Artifacts
none

## Results
n/a

## Claim gate
Supports (descriptive): n/a
Does not support: n/a

## Next steps
- W0.1: integrate.
"""

D14_PATH = "scripts/demo/research/devlog/2026/2026-10-02_d14-overlap.md"
ADOPT_PATH = "scripts/demo/research/devlog/2026/2026-10-08_spine-adopted.md"


def put(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_study(root: Path) -> Path:
    research = "scripts/demo/research/"
    put(root, research + "study_config.json", STUDY_CONFIG)
    put(root, research + "THEME.md", THEME)
    put(root, research + "hypotheses.md", HYPOTHESES)
    put(root, research + "alternatives.csv", ALTERNATIVES)
    put(root, research + "analyses.csv", ANALYSES)
    put(root, research + "campaigns.csv", CAMPAIGNS)
    put(root, research + "asks/2026-10-08_team-report.md", ASKS)
    put(root, "scripts/demo/workstreams/W0_context/README.md", WS0)
    put(root, "scripts/demo/workstreams/W1_aggregation/README.md", WS1)
    (root / research / "devlog").mkdir(parents=True, exist_ok=True)
    shutil.copy(str(TEMPLATE), str(root / research / "devlog" / "_TEMPLATE.md"))
    put(root, D14_PATH, D14_ENTRY)
    put(root, ADOPT_PATH, ADOPT_ENTRY)
    put(root, "scripts/demo/notebooks/atlas.ipynb", "{}\n")
    put(root, "scripts/demo/notebooks/overlap.ipynb", "{}\n")
    put(root, "scripts/demo/code/overlap.py", "# demo\n")
    return root


def edit(root: Path, rel: str, old: str, new: str) -> None:
    path = root / rel
    text = path.read_text(encoding="utf-8")
    assert old in text, "mutation anchor %r not found in %s" % (old, rel)
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def run(root: Path, *args: str):
    argv = list(args)
    if not argv or argv[0] in ("check", "write"):
        argv = (argv or ["check"]) + ["--today", TODAY]
    return ri.main(argv, repo=root)


def load(root: Path):
    st = ri.Study(root, "demo", ri.Report(root)).load()
    return st, ri.Derived(st)


@pytest.fixture(autouse=True)
def _no_sc_env(monkeypatch):
    for var in ("SC_REPO_ROOT", "SC_EXPERIMENT_ROOT", "SC_EXPERIMENT_ID", "SC_DATA_ROOT", "SC_COHORT_H5AD"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def study(tmp_path):
    root = make_study(tmp_path / "repo")
    assert run(root, "write") == 0
    return root


# ---------------------------------------------------------------------------------------
# grammar
# ---------------------------------------------------------------------------------------


def test_frontmatter_subset(tmp_path):
    rep = ri.Report(tmp_path)
    lines = ["---", "id: x", "asks: [A-1, B-2 ]", "empty: []", "flag: true", "off: false",
             "commit: b60a6ee  # retro note", "title: 'quoted: yes'", "url: a#b", "---", "body"]
    fm = ri.parse_front_matter(lines, "f.md", rep)
    assert fm.ok and not rep.issues
    assert fm.data == {"id": "x", "asks": ["A-1", "B-2"], "empty": [], "flag": True, "off": False,
                       "commit": "b60a6ee", "title": "quoted: yes", "url": "a#b"}
    assert fm.body_start == 10 and fm.lines["commit"] == 7

    bad_cases = [
        (["---", "a: 1", "nested:", "  child: 2", "---"], 4),
        (["---", "items:", "- a", "---"], 3),
        (["---", "text: |", "---"], 2),
        (["---", "text: >", "---"], 2),
        (["---", "map: {a: 1}", "---"], 2),
        (["---", "a: 1", "a: 2", "---"], 3),
        (["---", "Bad Key: 1", "---"], 2),
        (["---", "a: 1"], 1),
        (["no front matter"], 1),
    ]
    for lines, lineno in bad_cases:
        rep = ri.Report(tmp_path)
        fm = ri.parse_front_matter(lines, "f.md", rep)
        assert not fm.ok, lines
        assert any(i.code == "E01" and i.line == lineno for i in rep.issues), (lines, rep.issues)
        assert all("devlog/README.md" in i.msg for i in rep.issues)


def test_tool_is_python38_syntax():
    ast.parse(TOOL_PATH.read_text(encoding="utf-8"), feature_version=(3, 8))


def test_no_studies_found(tmp_path, capsys):
    assert ri.main(["check"], repo=tmp_path) == 0
    assert "no studies found" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------
# clean fixture, determinism and error codes
# ---------------------------------------------------------------------------------------


def test_fixture_clean_and_deterministic(study, capsys):
    outputs = [study / "scripts/demo/research/STATUS.md", study / "scripts/demo/research/CLAIMS.md",
               study / "scripts/demo/research/devlog/INDEX.md"]
    first = [p.read_bytes() for p in outputs]
    for path in outputs:
        path.unlink()
    assert run(study, "write") == 0
    assert [p.read_bytes() for p in outputs] == first
    assert run(study, "check") == 0
    out = capsys.readouterr().out
    assert "check: 0 error(s), 0 stale file(s)" in out
    for blob in first:
        text = blob.decode("utf-8")
        assert text.startswith(ri.GENERATED_HEADER + "\n")
        assert "\r" not in text
        assert "(as of 2026-10-08)" in text
    status = first[0].decode("utf-8")
    assert "| P02 | 211 | 1462 | 14.4% | yes |" in status
    assert "Next free D ID: D17" in status
    claims = first[1].decode("utf-8")
    assert "**supports** Calls: 45 aggregate_shared of 47 representations." in claims
    assert "  - median 0.843" in claims
    assert "| H-S1-04 | supported_descriptive | ALT-PREV | partial |" in claims


def _set_compute_screen(root):
    edit(root, D14_PATH, "compute: notebook-cpu", "compute: screen-cpu")


MUTATIONS = {
    "E01": lambda r: edit(r, D14_PATH, "title: D14", "  nested: x\ntitle: D14"),
    "E02": lambda r: edit(r, D14_PATH, "type: analysis", "type: experiment"),
    "E03": lambda r: edit(r, D14_PATH, "date: 2026-10-02", "date: 2026-10-03"),
    "E04": lambda r: edit(r, D14_PATH, "closes: []", "closes: [W1.9]"),
    "E05": lambda r: edit(r, "scripts/demo/research/asks/2026-10-08_team-report.md",
                          "- [A-20261008-03]", "- [A-20261008-01] \"Again.\" -> D14\n- [A-20261008-03]"),
    "E06": lambda r: edit(r, D14_PATH, "## Why it matters\n", ""),
    "E07": lambda r: edit(r, "scripts/demo/research/THEME.md", "## 0. Working thesis",
                          "Data lives in " + HOME_NEEDLE + "someone/data.\n\n## 0. Working thesis"),
    "E08": lambda r: edit(r, D14_PATH, "  - calls.csv", "  - calls.csv\n- ../outside/table.csv"),
    "E09": lambda r: edit(r, D14_PATH, "commit: b60a6ee", "commit: abc"),
    "E10": lambda r: edit(r, D14_PATH, "compute: notebook-cpu\n", ""),
    "E11": lambda r: edit(r, ADOPT_PATH, "## Artifacts\nnone",
                          "## Artifacts\n- analysis/scorecards/stage2_diagnostic/new_slug/"),
    "E12": lambda r: (edit(r, D14_PATH, "closes: []", "closes: [W0.1]"),
                      edit(r, ADOPT_PATH, "closes: []", "closes: [W0.1]")),
    "E13": lambda r: edit(r, ADOPT_PATH, "H-S1-04=supported_descriptive", "H-S1-04=supported_publication"),
    "E14": lambda r: edit(r, ADOPT_PATH, ", H-S2-07=retired", ""),
    "E15": lambda r: edit(r, D14_PATH, "- IFN-alpha", "TODO: add the IFN-alpha line\n- IFN-alpha"),
    "E16": lambda r: edit(r, "scripts/demo/research/analyses.csv", "2026-10-08,planned", "2026-10-08,bogus"),
    "E17": lambda r: edit(r, "scripts/demo/research/campaigns.csv", "complete,never", "done,never"),
    "E18": lambda r: edit(r, "scripts/demo/workstreams/W1_aggregation/README.md", "id: W1", "id: W2"),
    "E19": lambda r: edit(r, "scripts/demo/research/alternatives.csv", "alt_id,name", "alt,name"),
    "E20": lambda r: edit(r, "scripts/demo/research/STATUS.md", "## 0. Cohort", "hand edit\n## 0. Cohort"),
}


@pytest.mark.parametrize("code", sorted(MUTATIONS))
def test_error_codes(study, capsys, code):
    capsys.readouterr()
    MUTATIONS[code](study)
    status = run(study, "check")
    out = capsys.readouterr().out
    codes = {line.split(" ", 1)[0] for line in out.splitlines() if re.match(r"^E\d\d ", line)}
    assert code in codes, out
    assert status == (3 if code == "E20" else 1), out
    if code == "E20":
        assert codes == {"E20"}
        assert "run python tools/research_index.py write" in out


def test_extra_entry_rules(study, capsys):
    # screen compute needs a literal screen line; backfilled global_selected C changes only warn (W11)
    _set_compute_screen(study)
    assert run(study, "check") == 1
    assert "E10 " + D14_PATH in capsys.readouterr().out
    edit(study, D14_PATH, "- Ran on CPU", "- screen -S W1_overlap, then the runner.\n- Ran on CPU")
    edit(study, D14_PATH, "c_policy: not_applicable\ntraining_n_changed: false",
         "c_policy: global_selected\ntraining_n_changed: true")
    run(study, "write")
    captured = capsys.readouterr()
    assert "W11 " + D14_PATH in captured.err
    # W11 records history on backfilled entries, so --strict does not promote it
    assert run(study, "check", "--strict") == 0
    assert "W11 " + D14_PATH in capsys.readouterr().err
    # protected Set A scorecards and another analysis's slug
    edit(study, ADOPT_PATH, "## Artifacts\nnone",
         "## Artifacts\n- analysis/scorecards/stage2_discovery_full_cohort_scorecard.csv\n"
         "- analysis/scorecards/stage2_diagnostic/patient_factor_overlap_v1/extra.csv")
    capsys.readouterr()
    assert run(study, "check") == 1
    out = capsys.readouterr().out
    assert "lists protected path" in out and "is the artifact_dir of D14" in out


def test_artifacts_strict(study, tmp_path, capsys):
    exp_root = tmp_path / "exps"
    (exp_root / "demo_exp" / "analysis/scorecards/stage2_diagnostic/patient_factor_overlap_v1").mkdir(parents=True)
    capsys.readouterr()
    assert run(study, "check", "--experiment-root", str(exp_root)) == 0
    err = capsys.readouterr().err
    assert "W10 scripts/demo/research/analyses.csv" in err and "findings.md" in err
    assert run(study, "check", "--experiment-root", str(exp_root), "--strict-artifacts") == 1
    (exp_root / "demo_exp" / "analysis/scorecards/stage2_diagnostic/patient_factor_overlap_v1/findings.md").write_text("x")
    for rel in ("analysis/scorecards/stage2_diagnostic/d1a_deep_dive", "analysis/scorecards/stage2_diagnostic/feedback_deep_dive"):
        (exp_root / "demo_exp" / rel).mkdir(parents=True)
    assert run(study, "check", "--experiment-root", str(exp_root), "--strict-artifacts") == 0


def test_strict_artifacts_fails_when_experiment_dir_unresolvable(study, tmp_path, capsys):
    empty = tmp_path / "no_experiments_here"
    empty.mkdir()
    capsys.readouterr()
    assert run(study, "check", "--experiment-root", str(empty)) == 0  # plain check skips artifact checks
    assert run(study, "check", "--experiment-root", str(empty), "--strict-artifacts") == 1
    out = capsys.readouterr().out
    assert "E21 " in out and "SC_EXPERIMENT_ROOT" in out and "SC_EXPERIMENT_ID" in out


def test_strict_promotes_other_warnings(study, capsys):
    _extra_entry(study, "2026-10-09_theme-change", {"verdicts": "[H-S1-04=open]"})
    run(study, "write")
    capsys.readouterr()
    assert run(study, "check") == 0
    assert "W09 " in capsys.readouterr().err
    assert run(study, "check", "--strict") == 1


def test_theme_queue_same_day_entries(study, capsys):
    # same date as THEME last_reviewed, verdict change, not cited by THEME: queued
    _extra_entry(study, "2026-10-08_zz-same-day", {"verdicts": "[H-S1-04=open]"})
    st, d = load(study)
    assert [e.id for e in d.theme_queue] == ["2026-10-08_zz-same-day"]
    # citing it in THEME clears it
    edit(study, "scripts/demo/research/THEME.md", "devlog 2026-10-08_spine-adopted",
         "devlog 2026-10-08_spine-adopted, 2026-10-08_zz-same-day")
    st, d = load(study)
    assert d.theme_queue == []
    # reviewed_through takes precedence: entries after it in INDEX order are queued
    edit(study, "scripts/demo/research/THEME.md", "last_reviewed: 2026-10-08",
         "last_reviewed: 2026-10-08\nreviewed_through: 2026-10-08_spine-adopted")
    st, d = load(study)
    assert [e.id for e in d.theme_queue] == ["2026-10-08_zz-same-day"]
    assert run(study, "write") == 0


def test_seq_breaks_same_day_ties(study):
    _extra_entry(study, "2026-10-09_a-final", {"verdicts": "[H-S1-04=open]"})
    _extra_entry(study, "2026-10-09_b-draft", {"verdicts": "[H-S1-04=not_supported]"})
    st, d = load(study)
    assert d.verdict("H-S1-04")[1] == "2026-10-09_b-draft"  # default: alphabetical by id
    edit(study, "scripts/demo/research/devlog/2026/2026-10-09_a-final.md", "supersedes: []", "supersedes: []\nseq: 1")
    st, d = load(study)
    assert d.verdict("H-S1-04") == ("2026-10-09", "2026-10-09_a-final", "open")
    assert [e.id for e in st.entries][-2:] == ["2026-10-09_b-draft", "2026-10-09_a-final"]
    edit(study, "scripts/demo/research/devlog/2026/2026-10-09_a-final.md", "seq: 1", "seq: first")
    assert run(study, "check") == 1


def test_e07_covers_new_tests_and_apps(study, capsys):
    put(study, "tests/test_factor_cards.py", "# data in " + HOME_NEEDLE + "someone\n")
    put(study, "scripts/mrd_stage0_2/apps/old34_factor_explorer.py", "ROOT = '" + HOME_NEEDLE + "someone'\n")
    capsys.readouterr()
    assert run(study, "check") == 1
    out = capsys.readouterr().out
    assert "E07 tests/test_factor_cards.py:1" in out
    assert "E07 scripts/mrd_stage0_2/apps/old34_factor_explorer.py:1" in out


# ---------------------------------------------------------------------------------------
# derivations
# ---------------------------------------------------------------------------------------


def test_task_state_derivation(study):
    st, d = load(study)
    assert d.task_state == {"W0.1": "in_progress", "W0.2": "open", "W1.1": "in_progress",
                            "W1.2": "open", "W1.3": "blocked"}
    assert st.tasks["W1.2"].depends == ["W1.1"] and st.tasks["W1.3"].blocked == "waiting for W2 bases"
    assert st.tasks["W1.1"].short == "D16 factor x patient matrix"
    edit(study, ADOPT_PATH, "closes: []", "closes: [W1.1, W1.3]")
    st, d = load(study)
    assert d.task_state["W1.1"] == "done" and d.task_state["W1.3"] == "done"
    status = ri.render_status(st, d)
    assert "W1.1 (met)" in status
    # a partial entry that closes a task only marks it in progress
    edit(study, ADOPT_PATH, "closes: [W1.1, W1.3]", "closes: [W0.2]")
    edit(study, ADOPT_PATH, "status: done", "status: partial")
    st, d = load(study)
    assert d.task_state["W0.2"] == "in_progress" and d.task_state["W1.1"] == "in_progress"
    assert "W1.1 (unmet)" in ri.render_status(st, d)


def _extra_entry(root, eid, fm_overrides, body_results="- n/a 1", supersedes="[]"):
    text = ADOPT_ENTRY.replace("id: 2026-10-08_spine-adopted", "id: " + eid)
    text = text.replace("date: 2026-10-08", "date: " + eid[:10])
    text = text.replace("asks: [A-20261008-02]", "asks: []")
    text = text.replace("verdicts: [H-S1-04=supported_descriptive, H-S2-07=retired]", "verdicts: []")
    text = text.replace("alts: [H-S1-04:ALT-PREV=partial]", "alts: []")
    text = text.replace("advances: [W0.1]", "advances: []")
    text = text.replace("supersedes: []", "supersedes: " + supersedes)
    text = text.replace("## Results\nn/a", "## Results\n" + body_results)
    for key, value in fm_overrides.items():
        text = re.sub(r"(?m)^%s: .*$" % key, "%s: %s" % (key, value), text)
    return put(root, "scripts/demo/research/devlog/%s/%s.md" % (eid[:4], eid), text)


def test_verdict_and_alt_latest_wins(study):
    _extra_entry(study, "2026-10-09_b-later", {"verdicts": "[H-S1-04=open]", "alts": "[H-S1-04:ALT-PREV=addressed]"})
    _extra_entry(study, "2026-10-09_a-earlier", {"verdicts": "[H-S1-04=not_supported]"})
    st, d = load(study)
    assert [v for _, _, v in d.verdict_hist["H-S1-04"]] == ["supported_descriptive", "supported_descriptive",
                                                             "not_supported", "open"]
    assert d.verdict("H-S1-04") == ("2026-10-09", "2026-10-09_b-later", "open")
    assert d.alt_state("H-S1-04", "ALT-PREV") == "addressed"
    assert d.alt_state("H-S1-04", "ALT-TRANS") == "open"
    assert d.verdict("H-S2-07")[2] == "retired"
    claims = ri.render_claims(st, d)
    assert "Verdict history: 2026-10-02 supported_descriptive" in claims
    assert "| H-S1-04 | open | ALT-PREV |" not in claims  # addressed alternatives leave the board


def test_evidence_tags_and_supersedes(study, capsys):
    st, d = load(study)
    evs = d.evidence["H-S1-04"]
    assert [(e.entry_id, e.n, e.role) for e in evs] == [("2026-10-02_d14-overlap", 1, "supports"),
                                                        ("2026-10-02_d14-overlap", 2, "qualifies")]
    assert evs[1].extra == ["  - median 0.843"]
    # a correction entry supersedes D14: its evidence leaves the current set, history keeps it
    _extra_entry(study, "2026-10-09_d14-correction", {"hypotheses": "[H-S1-04]"},
                 body_results="- [H-S1-04 qualifies] Overlap recomputed with 200 permutations.",
                 supersedes="[2026-10-02_d14-overlap]")
    st, d = load(study)
    assert [e.entry_id for e in d.evidence["H-S1-04"]] == ["2026-10-09_d14-correction"]
    assert len(d.evidence_all["H-S1-04"]) == 3
    assert "2026-10-02_d14-overlap" in d.superseded
    assert run(study, "write") == 1
    assert "E14 scripts/demo/research/hypotheses.md" in capsys.readouterr().out
    # a malformed tag is E03; superseding a later entry is E04
    edit(study, D14_PATH, "- IFN-alpha", "- [H-S1-04 support] bad tag 1\n- IFN-alpha")
    edit(study, D14_PATH, "supersedes: []", "supersedes: [2026-10-08_spine-adopted]")
    run(study, "check")
    out = capsys.readouterr().out
    assert "E03 " + D14_PATH in out and "is not an earlier entry" in out


def test_ask_state_rules(study):
    st, d = load(study)
    assert st.asks["A-20261008-02"].asker == "teammate:Ankit"
    assert st.asks["A-20261008-02"].text == '"Is the gap -> just L1 at fixed C?"'
    assert st.asks["A-20261008-02"].routes == ["W1.1"]
    # cited by a done analysis entry, routed task still open -> partial
    assert d.ask_state["A-20261008-01"] == "partial"
    # cited by a done decision entry, routed W1.1 in progress -> partial
    assert d.ask_state["A-20261008-02"] == "partial"
    assert d.ask_state["A-20261008-03"] == "open"
    _extra_entry(study, "2026-10-09_meeting", {"type": "meeting", "asks": "[A-20261008-03]"})
    st, d = load(study)
    assert d.ask_state["A-20261008-03"] == "open"  # meetings record asks, never answer them
    _extra_entry(study, "2026-10-10_theme-draft", {"type": "docs", "asks": "[A-20261008-03]", "status": "partial"})
    st, d = load(study)
    assert d.ask_state["A-20261008-03"] == "partial"
    _extra_entry(study, "2026-10-11_theme-done", {"type": "docs", "asks": "[A-20261008-03]", "closes": "[W0.2]"})
    st, d = load(study)
    assert d.ask_state["A-20261008-03"] == "answered"
    assert "A-20261008-03" not in ri.render_status(st, d)


# ---------------------------------------------------------------------------------------
# scaffolding and queries
# ---------------------------------------------------------------------------------------


def test_reserve_id_next(study, capsys):
    rows = [("D%d" % n, "D") for n in range(1, 16)] + [("D15b", "D")]
    rows += [("D1DD%d" % n, "D1DD") for n in range(1, 14)] + [("F%d" % n, "F") for n in range(1, 10)]
    lines = [ANALYSES_HEADER.rstrip("\n")]
    for aid, fam in rows:
        lines.append("%s,%s,Title %s,Question?,Q1,user,,W1,scripts/demo/notebooks/atlas.ipynb,,demo_exp,,,%s,2026-09-01,historical,"
                     % (aid, fam, aid, aid))
        if aid == "D15b":
            lines += ["D%d,D,Planned %d,Question?,Q1,user,,W1,,,demo_exp,,,D%d,2026-10-08,planned," % (n, n, n) for n in range(16, 26)]
    put(study, "scripts/demo/research/analyses.csv", "\n".join(lines) + "\n")
    st, d = load(study)
    assert d.next_d == "D26"
    capsys.readouterr()
    assert ri.main(["reserve-id", "--workstream", "W1", "--title", "Null model, matched sets", "--questions", "Q2;Q1",
                    "--slug", "null_model1_v1", "--date", "2026-10-09"], repo=study) == 0
    assert capsys.readouterr().out.strip() == "D26"
    text = (study / "scripts/demo/research/analyses.csv").read_text()
    ids = [line.split(",", 1)[0] for line in text.splitlines()[1:]]
    assert ids.index("D26") == ids.index("D25") + 1 and ids.index("D26") < ids.index("D1DD1")
    assert ('D26,D,"Null model, matched sets",,Q2;Q1,not_recorded,,W1,,,demo_exp,'
            "analysis/scorecards/stage2_diagnostic/null_model1_v1,,D26,2026-10-09,planned,") in text
    st, d = load(study)
    assert st.analyses["D26"]["status"] == "planned" and d.next_d == "D27"
    assert ri.main(["reserve-id", "--workstream", "W1", "--title", "x", "--slug", "unversioned"], repo=study) == 2
    assert ri.main(["reserve-id", "--workstream", "W7", "--title", "x"], repo=study) == 1
    # asks, code and findings_md can be filled at reservation time
    capsys.readouterr()
    assert ri.main(["reserve-id", "--workstream", "W1", "--title", "ADT associations", "--questions", "QI",
                    "--slug", "factor_adt_associations_v1", "--asks", "A-20261008-01",
                    "--code", "scripts/demo/code/run_adt.py", "--findings-md", "findings.md", "--date", "2026-10-09"],
                   repo=study) == 0
    assert capsys.readouterr().out.strip() == "D27"
    text = (study / "scripts/demo/research/analyses.csv").read_text()
    assert ("D27,D,ADT associations,,QI,not_recorded,A-20261008-01,W1,,scripts/demo/code/run_adt.py,demo_exp,"
            "analysis/scorecards/stage2_diagnostic/factor_adt_associations_v1,"
            "analysis/scorecards/stage2_diagnostic/factor_adt_associations_v1/findings.md,D27,2026-10-09,planned,") in text
    assert ri.main(["reserve-id", "--workstream", "W1", "--title", "x", "--asks", "A-20990101-01"], repo=study) == 1
    st, d = load(study)
    assert st.analyses["D27"]["asks"] == "A-20261008-01" and st.analyses["D27"]["code"] == "scripts/demo/code/run_adt.py"


def test_new_entry_scaffold_fails_until_filled(study, capsys):
    capsys.readouterr()
    args = ["new-entry", "scaffold-test", "--workstream", "W1", "--analyses", "D14", "--asks", "A-20261008-01",
            "--date", "2026-10-09"]
    assert ri.main(args, repo=study) == 0
    rel = capsys.readouterr().out.strip()
    assert rel == "scripts/demo/research/devlog/2026/2026-10-09_scaffold-test.md"
    path = study / rel
    text = path.read_text()
    for line in ("id: 2026-10-09_scaffold-test", "date: 2026-10-09", "type: analysis", "workstream: W1",
                 "analyses: [D14]", "asks: [A-20261008-01]", "experiment: demo_exp", "commit: pending",
                 "backfilled: false", "date_source: authored"):
        assert line + "\n" in text
    assert "compute: TODO:" in text and "title: TODO:" in text
    assert ri.main(args, repo=study) == 1  # append-only: never overwrite
    assert run(study, "write") == 1
    assert "E15 " + rel in capsys.readouterr().out
    fills = {"title": "Scaffold test", "asked_by": "user", "implemented_by": "claude-code", "questions": "[Q1]",
             "hypotheses": "[]", "verdicts": "[]", "alts": "[]", "closes": "[]", "advances": "[W1.2]",
             "claim_level": "descriptive", "compute": "agent-cpu", "c_policy": "not_applicable",
             "training_n_changed": "false", "status": "done", "supersedes": "[]", "run_head": "b60a6ee",
             "dirty_run": "true"}
    out = []
    for line in path.read_text().splitlines():
        key = line.split(":", 1)[0]
        if key in fills and "TODO:" in line:
            line = "%s: %s" % (key, fills[key])
        out.append(re.sub(r"TODO: .*", "filled with 1 number", line))
    path.write_text("\n".join(out) + "\n")
    assert run(study, "write") == 0
    # a non-analysis scaffold drops the compute keys
    assert ri.main(["new-entry", "a-decision", "--workstream", "W0", "--type", "decision", "--date", "2026-10-09"], repo=study) == 0
    decision = (study / "scripts/demo/research/devlog/2026/2026-10-09_a-decision.md").read_text()
    assert "compute:" not in decision and "c_policy:" not in decision
    assert ri.main(["new-entry", "Bad_Slug", "--workstream", "W0"], repo=study) == 2
    assert ri.main(["new-entry", "ok", "--workstream", "W9"], repo=study) == 1


@pytest.mark.parametrize("ident, needle", [
    ("D14", "Per-patient factor-score overlap"), ("H-S1-04", "Verdict: supported_descriptive"),
    ("W1", "W1.3 (P1, blocked)"), ("W1.2", "depends: W1.1 (unmet)"), ("A-20261008-01", "2026-10-02_d14-overlap"),
    ("ALT-PREV", "H-S1-04 (supported_descriptive): partial"), ("2026-10-02_d14-overlap", "H-S1-04 supports"),
])
def test_show(study, capsys, ident, needle):
    capsys.readouterr()
    assert ri.main(["show", ident], repo=study) == 0
    assert needle in capsys.readouterr().out


def test_show_unknown_and_portability(study, capsys):
    capsys.readouterr()
    assert ri.main(["show", "D41"], repo=study) == 1
    assert "did you mean" in capsys.readouterr().err
    put(study, "scripts/demo/legacy_runner.py", "ROOT = '" + HOME_NEEDLE + "someone/repo'\n")
    put(study, "scripts/demo/research/notes.json", "{}\n")
    assert ri.main(["portability"], repo=study) == 0
    out = capsys.readouterr().out
    assert "scripts/demo/legacy_runner.py  1 line(s)" in out and "notes.json" not in out


# ---------------------------------------------------------------------------------------
# real repo (skipped until the spine exists)
# ---------------------------------------------------------------------------------------


@pytest.mark.skipif(not (REPO / "scripts/mrd_stage0_2/research/study_config.json").is_file(),
                    reason="research spine not present")
def test_real_repo_check(capsys):
    status = ri.main(["check"], repo=REPO)
    out = capsys.readouterr().out
    assert status == 0, out
