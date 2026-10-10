#!/usr/bin/env bash
# Build the 2026-10-08 questions report: regenerate the figures, then compile the PDF(s).
#
#   bash build.sh                 # default: figures + report_core.pdf (the short read, at most 12 pages)
#   bash build.sh --full          # figures + report_full.pdf (the long companion, every section in full)
#   bash build.sh --team          # figures + report_full_team.pdf (the long companion without Appendix C)
#   bash build.sh --all           # figures + all three PDFs
#   bash build.sh --no-figures    # reuse figures/*.pdf (combine with any of the above)
#   bash build.sh --figures 03 07 # regenerate only these sections' figures, then the PDF(s)
#
# Sources: report_core.tex + sections_core/*.tex (core); report.tex + sections/*.tex (full and team).
# Python: $SC_PYTHON if set, else the lab conda env (sc-classification-rapids-py311) under $HOME,
# else the python on PATH. LaTeX auxiliary files go to a temporary build dir (kept out of the repo, so
# research_index.py check never lints a .log); set REPORT_BUILD_DIR to keep them somewhere fixed.
# Only the PDFs are copied back next to the sources. figures/*.{pdf,png} and the full/team PDFs are gitignored;
# report_core.pdf is not (a .gitignore exception): rebuild it and commit it with its .tex sources.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

run_figures=1
targets=()
fig_sections=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-figures) run_figures=0; shift ;;
    --core) targets+=(core); shift ;;
    --full) targets+=(full); shift ;;
    --team) targets+=(team); shift ;;
    --all) targets+=(core full team); shift ;;
    --figures) shift; while [[ $# -gt 0 && "$1" != --* ]]; do fig_sections+=("$1"); shift; done ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ ${#targets[@]} -eq 0 ]] && targets=(core)

LAB_ENV="$HOME/miniconda3/envs/sc-classification-rapids-py311"
if [[ -n "${SC_PYTHON:-}" ]]; then
  PY="$SC_PYTHON"
elif [[ -x "$LAB_ENV/bin/python" ]]; then
  PY="$LAB_ENV/bin/python"
  export CONDA_PREFIX="${CONDA_PREFIX:-$LAB_ENV}"
else
  PY="python"
fi

# Shared lab server: keep BLAS threads modest (figure code is light).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

if [[ "$run_figures" -eq 1 ]]; then
  echo "== figures ($PY)"
  "$PY" make_figures.py ${fig_sections[@]+"${fig_sections[@]}"}
fi

command -v pdflatex >/dev/null || { echo "pdflatex not found on PATH" >&2; exit 1; }
BUILD_DIR="${REPORT_BUILD_DIR:-$(mktemp -d -t questions_report.XXXXXX)}"
mkdir -p "$BUILD_DIR"

# compile JOB TEX_INPUT: pdflatex until cross-references settle, then copy JOB.pdf here and report.
compile() {
  local job="$1" tex_input="$2" pass pages overfull undefined
  echo "== pdflatex $job (aux files in $BUILD_DIR)"
  # Two passes (cross-references, page numbers), plus a third if LaTeX asks for a rerun.
  for pass in 1 2 3; do
    if [[ "$pass" -eq 3 ]] && ! grep -q 'Rerun to get' "$BUILD_DIR/$job.log"; then
      break
    fi
    if ! pdflatex -interaction=nonstopmode -file-line-error -jobname="$job" -output-directory "$BUILD_DIR" "$tex_input" >/dev/null; then
      echo "pdflatex pass $pass failed; see $BUILD_DIR/$job.log" >&2
      grep -n -E -A3 '^!|\.tex:[0-9]+: ' "$BUILD_DIR/$job.log" | head -40 >&2 || true
      exit 1
    fi
  done
  cp "$BUILD_DIR/$job.pdf" "$HERE/$job.pdf"
  if command -v pdfinfo >/dev/null; then
    pages="$(pdfinfo "$HERE/$job.pdf" | awk '/^Pages:/ {print $2}')"
  else
    pages="$(grep -o 'Output written on .*(\([0-9]*\) pages' "$BUILD_DIR/$job.log" | grep -o '([0-9]* pages' | tr -dc '0-9' || true)"
  fi
  overfull="$(grep -c '^Overfull' "$BUILD_DIR/$job.log" || true)"
  undefined="$(grep -c 'undefined' "$BUILD_DIR/$job.log" || true)"
  echo "== done: $job.pdf, ${pages:-?} pages; overfull boxes: $overfull; lines mentioning 'undefined': $undefined"
  if [[ "$job" == "report_core" && -n "${pages:-}" && "$pages" -gt 12 ]]; then
    echo "WARNING: report_core.pdf has $pages pages; its limit is 12 (see README.md)" >&2
  fi
}

for target in "${targets[@]}"; do
  case "$target" in
    core) compile report_core report_core.tex ;;
    full) compile report_full report.tex ;;
    # The team build defines \TEAMBUILD before report.tex, which leaves out Appendix C (for you only).
    team) compile report_full_team '\def\TEAMBUILD{1}\input{report.tex}' ;;
  esac
done
echo "log dir: $BUILD_DIR"
