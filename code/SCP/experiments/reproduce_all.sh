# run all simulations sequentially
# this will take some time

cd "$(dirname "$0")/.."  # cd to repo root.
mkdir -p model
mkdir -p results
set +x

printf "\n=============== Runnung: Fig3.sh ===============\n\n"
bash experiments/Fig3.sh > experiments/Fig3.txt 2>&1

printf "\n=============== Runnung: Fig4.sh ===============\n\n"
bash experiments/Fig4.sh > experiments/Fig4.txt 2>&1

printf "\n=============== Runnung: Fig5.sh ===============\n\n"
bash experiments/Fig5.sh > experiments/Fig5.txt 2>&1

printf "\n=============== Runnung: Fig_app.sh ===============\n\n"
bash experiments/Fig_app.sh > experiments/Fig_app.txt 2>&1
