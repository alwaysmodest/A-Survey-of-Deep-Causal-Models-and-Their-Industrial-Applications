# SyncTwin: Treatment Effect Estimation with Longitudinal Outcomes (NeurIPS 2021)

SyncTwin is a treatment effect estimation method tailored for observational studies with longitudinal data.
Specifically, it applies to the *LIP* setting: Longitudinal, Irregular and Point treatment.
In these studies, the covariates are observed at irregular intervals leading to treatment allocation;
the outcomes are measured longitudinally before and after the treatment; 
the treatment is assigned at a specific time point and stays unchanged during the study. 

The key insight of SyncTwin is to fully leverage the pre-treatment outcomes. 
It uses the temporal structure in the outcome time series to *improve the accuracy of* counterfactual prediction.
It further uses the pre-treatment outcomes to *control the estimation error* on the individual level.
Finally, the method enables *interpretability by example*: the user can examine the key contributing examples that leads to the estimate. 



## Installation

To run the code locally, make sure to first install the required python packages specified in `requirements.txt`. Python 3.7 is recommended for best compatibility. Note that `tensorflow` and `GPy` are only needed for running the benchmarks. The directory `clairvoyance` contains a streamlined version of the [clairvoyance](https://github.com/vanderschaarlab/clairvoyance) library. It is used to run the benchmarks CRN and RMSN.

For some benchmarks (SC, MC-NNM, 1NN), we use their public implementations in the [R](https://www.r-project.org/) language. To run these benchmarks, please install R and the dependencies listed in `requirements_R.txt`.

For coda users, an [environment YAML](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-from-file) file [`environment.yml`](./environment.yml) is provided, which includes both Python and R dependencies.



## Usage 

Scripts for reproducing paper experiments are provided under the directory [`experiments/`](experiments/).

The `reproduce_all.sh` shell script contains commands to reproduce *all* tables and figures in the paper.
The `Fig[x].sh` or `Tab[x].sh`  shell script contain commands to generate results for individual figures or tables.
The `Fig[x].ipynb` notebooks contain commands to create the visualizations. 
The results will be written in the `results` folder. For instance, `Tab2_C1_MAE.txt` corresponds to the first Column of Table 2.

An implementation of SyncTwin is provided in the file `SyncTwin.py`.
Note that SyncTwin is a general framework agnostic to the exact architectural choice of encoder and decoder.
In this implementation, we use attentive GRU-D encoder and time-LSTM decoder.
In the simulations, SyncTwin is trained in  `pkpd_sim3_model_training.py`.



## Citation

If you find the software useful, please consider citing the following [paper](https://proceedings.neurips.cc/paper/2021/hash/19485224d128528da1602ca47383f078-Abstract.html):

```
@inproceedings{synctwin2021,
  title={SyncTwin: Treatment Effect Estimation with Longitudinal Outcomes},
  author={Qian, Zhaozhi and Zhang, Yao and Bica, Ioana and Wood, Angela and van der Schaar, Mihaela},
  booktitle={Advances in neural information processing systems},
  year={2021}
}
```



## License
Copyright 2021, Zhaozhi Qian.

This software is released under the MIT license.
