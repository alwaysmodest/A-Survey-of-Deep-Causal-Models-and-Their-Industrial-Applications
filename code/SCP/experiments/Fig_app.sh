cd "$(dirname "$0")/.."  # cd to repo root.
mkdir -p model
mkdir -p results
set +x


prefix=2021

for linear in linear nonlinear
do
    for setting in  confounding_level p_confounder_cause p_cause_cause cause_noise
    do
        set -x
        bash experiments/run_one_simulation.sh ${setting} ${linear} ${prefix}
        { set +x; } 2>/dev/null
    done
done


for linear in linear nonlinear
do
    for setting in confounding_level p_confounder_cause p_cause_cause cause_noise
    do
        set -x
        bash experiments/summarize_one_simulation.sh ${setting} ${linear} ${prefix}
        { set +x; } 2>/dev/null
    done
done


for level in 2 4 6 8 10
do
    for seed in 1 2 3 4 5
    do
        set -x
        python -u -m scp.run_simulation_scp --config=ea_balance_${level} --seed=$seed > results/ea_scp_${level}_$seed.txt
        python -u -m benchmarks.run_simulation_dor --config=ea_balance_${level} --ablation=dor-perturb_subset-0 --seed=$seed > results/ea_dor_${level}_$seed.txt
        { set +x; } 2>/dev/null
    done
done

rm -f results/results_ea_confound.txt
for level in 2 4 6 8 10
do
    for seed in 1 2 3 4 5
    do
    value=`tail -n 1 results/ea_scp_${level}_$seed.txt`
    echo "SCP ${level} ${value} $seed" >> results/results_ea_confound.txt

    value=`tail -n 1 results/ea_dor_${level}_$seed.txt`
    echo "DOR ${level} ${value} $seed" >> results/results_ea_confound.txt
    done
done

cat results/results_ea_confound.txt
