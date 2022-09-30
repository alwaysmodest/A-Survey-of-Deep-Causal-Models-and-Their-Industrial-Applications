import argparse

import sim_config
from benchmarks import (
    run_simulation_bmc,
    run_simulation_dor,
    run_simulation_drcrn,
    run_simulation_overlap,
    run_simulation_propensity,
    run_simulation_tarnet,
    run_simulation_vsr,
)
from scp import run_simulation_scp, run_simulation_scp_nn

parser = argparse.ArgumentParser("PKPD simulation")
parser.add_argument("--method", type=str)
parser.add_argument("--config", type=str)
parser.add_argument("--eval_only", choices=["True", "False"], default="False", type=str)
parser.add_argument("--eval_delta", choices=["True", "False"], default="False", type=str)

args = parser.parse_args()
method = args.method
config_key = args.config
eval_only = args.eval_only == "True"
eval_delta = args.eval_delta == "True"

try:
    config = sim_config.sim_dict[config_key]
except Exception:  # pylint: disable=broad-except
    print("Config {} is not found.".format(config_key))
    exit(-1)

if method == "scp":
    run_simulation_scp.run(config, eval_only=eval_only, eval_delta=eval_delta)
elif method == "dor":
    run_simulation_dor.run(config, eval_only=eval_only, eval_delta=eval_delta)
elif method == "propensity":
    run_simulation_propensity.run(config, eval_only=eval_only, eval_delta=eval_delta)
elif method == "tarnet":
    run_simulation_tarnet.run(config, eval_only=eval_only, eval_delta=eval_delta)
elif method == "vsr":
    run_simulation_vsr.run(config, eval_only=eval_only, eval_delta=eval_delta)
elif method == "drcrn":
    run_simulation_drcrn.run(config, eval_only=eval_only, eval_delta=eval_delta)
elif method == "overlap":
    run_simulation_overlap.run(config, eval_only=eval_only, eval_delta=eval_delta)
elif method == "bmc":
    run_simulation_bmc.run(config, eval_only=eval_only, eval_delta=eval_delta)
elif method == "scp_nn":
    run_simulation_scp_nn.run(config, eval_only=eval_only, eval_delta=eval_delta)
