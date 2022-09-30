### omitted variable experiment

cd "$(dirname "$0")/.."  # cd to repo root.
mkdir -p models
mkdir -p results
set +x



# one omitted variable
set -x
python -u -m experiments.pkpd_sim3_data_generation --sim_id=sync1-p50-h1 --train_step=25 --step=30 --seed=100 --hidden_confounder=1
bash experiments/run_one_sim_bias.sh sync1-p50-h1 1.5
bash experiments/summarize_one_sim.sh sync1-p50-h1


# more omitted variables
python -u -m experiments.pkpd_sim3_data_generation --sim_id=sync1-p50-h2 --train_step=25 --step=30 --seed=100 --hidden_confounder=2
bash experiments/run_one_sim_bias.sh sync1-p50-h2 1.5
bash experiments/summarize_one_sim.sh sync1-p50-h2
set +x
