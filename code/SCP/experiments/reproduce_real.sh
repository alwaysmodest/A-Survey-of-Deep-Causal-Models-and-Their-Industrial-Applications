cd "$(dirname "$0")/.."  # cd to repo root.
mkdir -p model
mkdir -p results
set +x

printf "\n=============== Runnung: run_one_real.sh ===============\n\n"
bash experiments/run_one_real.sh > experiments/reproduce_real.txt 2>&1

printf "\n=============== Runnung: summarise_real.sh ===============\n\n"
bash experiments/summarise_real.sh > experiments/reproduce_real.txt 2>&1
