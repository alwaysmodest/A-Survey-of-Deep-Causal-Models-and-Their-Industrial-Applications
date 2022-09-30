# the first script to run

cd "$(dirname "$0")/.."  # cd to repo root.
mkdir -p models
mkdir -p results
set +x



set -x
bash experiments/run_one_sim.sh sync1 25 30 200 200 1
bash experiments/run_one_sim.sh sync2 15 20 200 200 1
bash experiments/run_one_sim.sh sync3 45 50 200 200 1

bash experiments/run_one_sim.sh sync6d 25 30 1000 200 1.5
bash experiments/run_one_sim.sh sync7d 15 20 1000 200 1.5
bash experiments/run_one_sim.sh sync8d 45 50 1000 200 1.5

bash experiments/summarize_one_sim.sh sync1
bash experiments/summarize_one_sim.sh sync2
bash experiments/summarize_one_sim.sh sync3

bash experiments/summarize_one_sim.sh sync6d
bash experiments/summarize_one_sim.sh sync7d
bash experiments/summarize_one_sim.sh sync8d
set +x

cp results/Tab2_C3_MAE.txt results/Tab6_C2_MAE.txt
cp results/Tab2_C6_MAE.txt results/Tab6_C5_MAE.txt
