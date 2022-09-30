# Run all (non-real data) experiments.

cd "$(dirname "$0")/.."  # cd to repo root.
mkdir -p models
mkdir -p results
set +x



printf "\n=============== Runnung: Tab6.sh ===============\n\n"
bash experiments/Tab6.sh > results/Tab6.log 2>&1

printf "\n=============== Runnung: Fig3.sh ===============\n\n"
bash experiments/Fig3.sh > results/Fig3.log 2>&1

printf "\n=============== Runnung: Tab2.sh ===============\n\n"
bash experiments/Tab2.sh > results/Tab2.log 2>&1

printf "\n=============== Runnung: Tab5.sh ===============\n\n"
bash experiments/Tab5.sh > results/Tab5.log 2>&1

printf "\n=============== Runnung: Tab7.sh ===============\n\n"
bash experiments/Tab7.sh > results/Tab7.log 2>&1
