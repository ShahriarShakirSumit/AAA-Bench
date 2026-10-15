#!/usr/bin/env bash
# Recompute every table, figure and number of the paper from the files in this repository.
# CPU only, a few minutes. Results land in results/.
set -euo pipefail
cd "$(dirname "$0")/.."

python reproduce/validate_deconfound.py
python reproduce/compute_agreement.py
python reproduce/analysis_human_ceiling.py
python reproduce/make_tables.py
python reproduce/analysis_stats.py
python reproduce/analysis_baselines.py
python reproduce/analysis_utility.py
python reproduce/analysis_errors.py
python reproduce/analysis_sensitivity.py
python reproduce/analysis_detector.py
python reproduce/parser_validation.py
python reproduce/analysis_same_side.py
python reproduce/make_figures.py
python reproduce/audit_claims.py
