# Old script.

rm -r -f model/*

# n_confounder_40_linear n_confounder_25_linear n_confounder_10_linear
# p_confounder_cause_0.1_linear p_confounder_cause_0.3_linear p_confounder_cause_0.5_linear n_cause_2_linear n_cause_5_linear n_cause_10_linear
# n_flip_1_linear n_flip_3_linear n_flip_5_linear
# p_cause_cause_0.1_linear p_cause_cause_0.3_linear p_cause_cause_0.5_linear p_cause_cause_0.8_linear p_cause_cause_1.0_linear

# confounding_level_1.0_${linear} confounding_level_3.0_${linear} confounding_level_5.0_${linear} confounding_level_7.0_${linear}
# n_flip_1_${linear} n_flip_3_${linear} n_flip_5_${linear}

for linear in linear nonlinear
do
    for config in  p_confounder_cause_0.1_${linear} p_confounder_cause_0.3_${linear} p_confounder_cause_0.5_${linear} n_cause_2_${linear} n_cause_5_${linear} n_cause_10_${linear} n_confounder_40_${linear} n_confounder_30_${linear} n_confounder_10_${linear} p_cause_cause_0.1_${linear} p_cause_cause_0.3_${linear} p_cause_cause_0.5_${linear} p_cause_cause_0.8_${linear}
    do
        # for method in dor scp propensity tarnet vsr
        for method in overlap bmc
        do
            python -u -m run_simulation --method=${method} --config=${config} > results/${method}_${config}.txt
        done

        # python -u -m run_simulation_scp --method=ablation  --config=${config} > results/scp_ablation_${config}.txt
    done
done

# config=n_confounder_40_linear
#
rm -f results/results_nips.txt
touch results/results_nips.txt

for linear in linear nonlinear
do
# for config in n_flip_1_linear n_flip_3_linear n_flip_5_linear n_confounder_40_linear n_confounder_25_linear n_confounder_10_linear p_confounder_cause_0.1_linear p_confounder_cause_0.3_linear p_confounder_cause_0.5_linear n_cause_2_linear n_cause_5_linear n_cause_10_linear
    for config in n_flip_1_${linear} n_flip_3_${linear} n_flip_5_${linear} p_confounder_cause_0.1_${linear} p_confounder_cause_0.3_${linear} p_confounder_cause_0.5_${linear} n_cause_2_${linear} n_cause_5_${linear} n_cause_10_${linear} n_confounder_40_${linear} n_confounder_25_${linear} n_confounder_10_${linear} p_cause_cause_0.1_${linear} p_cause_cause_0.3_${linear} p_cause_cause_0.5_${linear} p_cause_cause_0.8_${linear} p_cause_cause_1.0_${linear}
    # for config in confounding_level_1.0_${linear} confounding_level_3.0_${linear} confounding_level_5.0_${linear} confounding_level_7.0_${linear}
    do
        for method in dor scp propensity tarnet vsr overlap bmc
        do
            value=`tail -n 1 results/${method}_${config}.txt`
            echo "${method} ${config} ${value}" >> results/results_nips.txt
        done
    done
done
cat results/results_nips.txt
