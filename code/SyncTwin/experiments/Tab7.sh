# run Tab6.sh before

cd "$(dirname "$0")/.."  # cd to repo root.
mkdir -p models
mkdir -p results
set +x



set -x
python -u -m experiments.interpretability --sim_id=sync1 > results/Tab7_C1.txt
python -u -m experiments.interpretability --sim_id=sync6d > results/Tab7_C4.txt
set +x
