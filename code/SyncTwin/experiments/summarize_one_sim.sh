cd "$(dirname "$0")/.."  # cd to repo root.
mkdir -p models
mkdir -p results
set +x



sim_id=$1
out_id=""

if [[ ${sim_id} == "sync1" ]]
then
    out_id=Tab2_C3
elif [[ ${sim_id} == "sync2" ]]
then
    out_id=Tab6_C1
elif [[ ${sim_id} == "sync3" ]]
then
    out_id=Tab6_C3
elif [[ ${sim_id} == "sync6d" ]]
then
    out_id=Tab2_C6
elif [[ ${sim_id} == "sync7d" ]]
then
    out_id=Tab6_C4
elif [[ ${sim_id} == "sync8d" ]]
then
    out_id=Tab6_C6
elif [[ ${sim_id} == "sync1-p10" ]]
then
    out_id=Tab2_C1
elif [[ ${sim_id} == "sync1-p25" ]]
then
    out_id=Tab2_C2
elif [[ ${sim_id} == "sync6d-p10" ]]
then
    out_id=Tab2_C4
elif [[ ${sim_id} == "sync6d-p25" ]]
then
    out_id=Tab2_C5
elif [[ ${sim_id} == "sync1-miss-0.3" ]]
then
    out_id=Tab5_C1
elif [[ ${sim_id} == "sync1-miss-0.5" ]]
then
    out_id=Tab5_C2
elif [[ ${sim_id} == "sync1-miss-0.7" ]]
then
    out_id=Tab5_C3
elif [[ ${sim_id} == "sync6d-miss-0.3" ]]
then
    out_id=Tab5_C4
elif [[ ${sim_id} == "sync6d-miss-0.5" ]]
then
    out_id=Tab5_C5
elif [[ ${sim_id} == "sync6d-miss-0.7" ]]
then
    out_id=Tab5_C6
elif [[ ${sim_id} == "sync1-p50-h1" ]]
then
    out_id=Fig3_C2
elif [[ ${sim_id} == "sync1-p50-h2" ]]
then
    out_id=Fig3_C3
else
    echo "Setting ${sim_id} is not Found"
fi


set -x
python -u -m experiments.synth_eval --sim_id=${sim_id} --seed=100 > models/${sim_id}-sc-sum.txt
set +x

rm -f results/${out_id}_MAE.txt
touch results/${out_id}_MAE.txt


echo SyncTwin >> results/${out_id}_MAE.txt
grep -E "Treatment effect MAE" models/${sim_id}-prognostic_linear.txt >> results/${out_id}_MAE.txt

echo SyncTwin-Lr >> results/${out_id}_MAE.txt
grep -E "Treatment effect MAE" models/${sim_id}-prognostic_none.txt >> results/${out_id}_MAE.txt

echo SyncTwin-Ls >> results/${out_id}_MAE.txt
grep -E "Treatment effect MAE" models/${sim_id}-prognostic_recon.txt >> results/${out_id}_MAE.txt

# Synthetic control
echo SC >> results/${out_id}_MAE.txt
grep -E "MAE" models/${sim_id}-sc-sum.txt  | head -n 1 >> results/${out_id}_MAE.txt

# Robust Synthetic control
echo RSC >> results/${out_id}_MAE.txt
tail models/${sim_id}-rsc.txt -n 1 >> results/${out_id}_MAE.txt

# Robust MC-NNM
echo MC-NNM >> results/${out_id}_MAE.txt
tail models/${sim_id}-nnm.txt -n 2 >> results/${out_id}_MAE.txt

# CFR-Net
echo CFR-Net >> results/${out_id}_MAE.txt
grep -E "MAE" models/${sim_id}_cfr.txt >> results/${out_id}_MAE.txt

# CRN
echo CRN >> results/${out_id}_MAE.txt
tail models/${sim_id}-CRN.txt -n 1 >> results/${out_id}_MAE.txt

# RMSN
echo RMSN >> results/${out_id}_MAE.txt
tail models/${sim_id}-RMSN.txt -n 1 >> results/${out_id}_MAE.txt

# CGP
echo CGP >> results/${out_id}_MAE.txt
grep -E "MAE" models/${sim_id}-gp.txt | head -n 6 | tail -n 1  >> results/${out_id}_MAE.txt

# 1NN
echo 1NN >> results/${out_id}_MAE.txt
grep -E "MAE" models/${sim_id}-sc-sum.txt  | tail -n 1 >> results/${out_id}_MAE.txt

cat results/${out_id}_MAE.txt
