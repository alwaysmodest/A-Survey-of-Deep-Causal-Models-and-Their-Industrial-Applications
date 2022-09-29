### Introduction
This project is the python implementation of the model "**A**dversarial **B**alancing based representation learning for **C**ausal **E**ffect **I**nference (**ABCEI**)"[https://arxiv.org/abs/1904.13335](https://arxiv.org/abs/1904.13335).
The entire framework is rewritten based on the [Counterfactual regression](https://github.com/clinicalml/cfrnet). Evaluation and Hyper-parameter search parts are reused to ensure the fairness of comparison.

### Requirements
To run the code, the following libraries are needed:  
>Python 3.5;  
>Tensorflow 1.4;  
>Numpy 1.15;

### Run
Three examples are provided for running this code on IHDP, Jobs and Twins datasets:
>ihdp.sh;  
>jobs.sh;  
>twins.sh;

Those files can be used combining with workload manager like slurm:  
>sbatch -p [cluster name] [--gres=gpu:1] ihdp.sh
