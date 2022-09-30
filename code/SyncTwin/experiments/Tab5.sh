
# irregular

cd "$(dirname "$0")/.."  # cd to repo root.
mkdir -p models
mkdir -p results
set +x



# generate data
set -x
python -u -m experiments.pkpd_sim3_irregular_generation --sim_id=sync1 --seed=100 --missing_pct=0.3
python -u -m experiments.pkpd_sim3_irregular_generation --sim_id=sync1 --seed=100 --missing_pct=0.5
python -u -m experiments.pkpd_sim3_irregular_generation --sim_id=sync1 --seed=100 --missing_pct=0.7


python -u -m experiments.pkpd_sim3_irregular_generation --sim_id=sync6d --seed=100 --missing_pct=0.3
python -u -m experiments.pkpd_sim3_irregular_generation --sim_id=sync6d --seed=100 --missing_pct=0.5
python -u -m experiments.pkpd_sim3_irregular_generation --sim_id=sync6d --seed=100 --missing_pct=0.7
set +x


# run benchmarks

bash experiments/run_one_sim_irregular.sh sync1-miss-0.3 1
bash experiments/run_one_sim_irregular.sh sync1-miss-0.5 1
bash experiments/run_one_sim_irregular.sh sync1-miss-0.7 1


bash experiments/run_one_sim_irregular.sh sync6d-miss-0.3 1.5
bash experiments/run_one_sim_irregular.sh sync6d-miss-0.5 1.5
bash experiments/run_one_sim_irregular.sh sync6d-miss-0.7 1.5

# ablate reconstruction loss

set -x
for sim_id in sync1-miss-0.3 sync1-miss-0.5 sync1-miss-0.7
do
    python -u -m experiments.pkpd_sim3_model_training --sim_id=${sim_id} --regular=False --seed=100 --model_id=-prognostic-recon --reduced_fine_tune=True --lam_recon=0 --pretrain_Y=True --itr=2 --tau=0.8 --linear_decoder=True  > models/${sim_id}-prognostic_recon.txt
done

for sim_id in sync6d-miss-0.3 sync6d-miss-0.5 sync6d-miss-0.7
do
    python -u -m experiments.pkpd_sim3_model_training --sim_id=${sim_id} --regular=False --seed=100 --model_id=-prognostic-recon --reduced_fine_tune=True --lam_recon=0 --pretrain_Y=True --itr=2 --linear_decoder=True  > models/${sim_id}-prognostic_recon.txt
done
set +x

# summarize results

bash experiments/summarize_one_sim.sh sync1-miss-0.3
bash experiments/summarize_one_sim.sh sync1-miss-0.5
bash experiments/summarize_one_sim.sh sync1-miss-0.7

bash experiments/summarize_one_sim.sh sync6d-miss-0.3
bash experiments/summarize_one_sim.sh sync6d-miss-0.5
bash experiments/summarize_one_sim.sh sync6d-miss-0.7
