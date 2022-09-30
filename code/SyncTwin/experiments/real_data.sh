# Run first:
#     * real_data_ml_preprocess.ipynb
#     * real_data_summary_stats.ipynb

cd "$(dirname "$0")/.."  # cd to repo root.
mkdir -p models
mkdir -p results
set +x



set -x
python -u -m experiments.real_data --model_id=m100-v3 --n_hidden=100 --itr_pretrain=15000 --itr=2 --fold=3 --data_version=3
python -u -m experiments.real_data_inference --model_id=m100-v3 --n_hidden=100 --itr_fine_tune=10000 --batch_size=1500 --data_version=3
set +x
