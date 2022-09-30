cd "$(dirname "$0")/.."  # cd to repo root.
mkdir -p model
mkdir -p results
set +x



config_arr=( 500 1000 1500 )


for config_val in "${config_arr[@]}"
do
    config="real_${config_val}"
    for method in dor scp propensity vsr tarnet drcrn bmc overlap
    do
        echo "results/${method}_${config}.txt"
        set -x
        python -u -m run_simulation --method=${method} --config=${config} > results2021/${method}_${config}.txt
        { set +x; } 2>/dev/null
    done
done
