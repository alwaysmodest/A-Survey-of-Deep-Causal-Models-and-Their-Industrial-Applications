# confounding_level, n_flip, p_confounder_cause, p_cause_cause n_cause, n_confounder

cd "$(dirname "$0")/.."  # cd to repo root.
mkdir -p model
mkdir -p results
set +x

data_setting=$1
linear_arg=$2
prefix=$3

mkdir -p results${prefix}

if [[ ${data_setting} == "confounding_level" ]]
then
    config_arr=( 1.0 3.0 5.0 7.0 )
elif [[ ${data_setting} == "n_flip" ]]
then
    config_arr=( 1 2 3 5 )
elif [[ ${data_setting} == "p_confounder_cause" ]]
then
    config_arr=( 0.1 0.3 0.5 0.8 )
elif [[ ${data_setting} == "p_cause_cause" ]]
then
    config_arr=( 0.1 0.3 0.5 0.8 )
elif [[ ${data_setting} == "p_outcome_single" ]]
then
    config_arr=( 0.1 0.3 0.5 0.8 )
elif [[ ${data_setting} == "p_outcome_double" ]]
then
    config_arr=( 0.05 0.1 0.15 0.2 )
elif [[ ${data_setting} == "cause_noise" ]]
then
    config_arr=( 0.01 0.3 0.5 0.8 )
elif [[ ${data_setting} == "sample_size_train" ]]
then
    config_arr=( 400 700 1400 2000 )
elif [[ ${data_setting} == "n_cause" ]]
then
    config_arr=( 2 5 7 10 )
elif [[ ${data_setting} == "n_confounder" ]]
then
    config_arr=( 10 20 30 40 )
else
    echo "Setting ${data_setting} is not Found"
fi


cp global_config.py results${prefix}/global_config.txt

for config_val in "${config_arr[@]}"
do
    config="${data_setting}_${config_val}_${linear_arg}"
    for method in dor scp propensity tarnet vsr drcrn bmc overlap
    do
        echo "results/${method}_${config}.txt"
        set -x
        python -u -m run_simulation --method=${method} --config=${config} > results${prefix}/${method}_${config}.txt
        { set +x; } 2>/dev/null
    done
done
