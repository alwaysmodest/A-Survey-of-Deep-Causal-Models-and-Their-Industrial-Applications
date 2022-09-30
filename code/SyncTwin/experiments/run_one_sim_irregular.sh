cd "$(dirname "$0")/.."  # cd to repo root.
mkdir -p models
mkdir -p results
set +x



sim_id=$1
tau=$2



set -x
python -u -m experiments.pkpd_sim3_model_training --regular=False --sim_id=${sim_id} --seed=100 --model_id=-prognostic-linear --reduced_fine_tune=True --tau=${tau} --lam_prognostic=1 --pretrain_Y=True --itr=3 --linear_decoder=True  > models/${sim_id}-prognostic_linear.txt
python -u -m experiments.pkpd_sim3_model_training --regular=False --sim_id=${sim_id} --seed=100 --model_id=-prognostic-none --reduced_fine_tune=True --tau=${tau} --lam_prognostic=0 --pretrain_Y=False --itr=3 --linear_decoder=True  > models/${sim_id}-prognostic_none.txt

Rscript experiments/pkpd_synth_control.R ${sim_id} 100 > models/${sim_id}-sc.txt
Rscript experiments/pkpd_1nn.R ${sim_id} 100 > models/${sim_id}-1nn.txt
Rscript experiments/pkpd-mc-nnm.R ${sim_id} 100 > models/${sim_id}-nnm.txt

python -u -m experiments.pkpd_sim3_cfr --sim_id=${sim_id} --seed=100 --lam_dist=0.1 > models/${sim_id}_cfr.txt
python -u -m experiments.clair_benchmark --sim_id=${sim_id} --seed=100 --model_name=CRN > models/${sim_id}-CRN.txt
python -u -m experiments.clair_benchmark --sim_id=${sim_id} --seed=100 --model_name=RMSN > models/${sim_id}-RMSN.txt
python -u -m experiments.gp_benchmark --sim_id=${sim_id}  > models/${sim_id}-gp.txt

# This takes a long time
python -u -m experiments.pkpd_rsc --sim_id=${sim_id} --seed=100 > models/${sim_id}-rsc.txt
set +x
