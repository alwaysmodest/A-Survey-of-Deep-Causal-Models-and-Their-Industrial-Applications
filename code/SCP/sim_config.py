from typing import NamedTuple

from global_config import (
    CAUSE_NOISE,
    CONFOUNDING_LEVEL,
    N_CAUSE,
    N_CONFOUNDER,
    OUTCOME_INTERACTION,
    OUTCOME_NOISE,
    P_CAUSE_CAUSE,
    P_CONFOUNDER_CAUSE,
    P_OUTCOME_DOUBLE,
    P_OUTCOME_SINGLE,
    SAMPLE_SIZE,
)
from sim_data_gen import DataGeneratorConfig

#
# ablation study
# P_CAUSE_CAUSE = 0.9

sim_dict = dict()

config = DataGeneratorConfig(
    n_confounder=17,
    n_cause=5,
    n_outcome=1,
    sample_size=3080,
    p_confounder_cause=0.3,
    p_cause_cause=0.0,
    cause_noise=1,
    outcome_noise=0.01,
    linear=True,
    n_flip=1,
    sim_id="real_3000",
    real_data=True,
)
sim_dict["real_3000"] = config

for sample_size in [500, 1000, 1500, 2000, 2500]:
    idd = "real_{}".format(sample_size)
    config = DataGeneratorConfig(
        n_confounder=17,
        n_cause=5,
        n_outcome=1,
        sample_size=sample_size,
        p_confounder_cause=0.3,
        p_cause_cause=P_CAUSE_CAUSE,
        cause_noise=1,
        outcome_noise=0.01,
        linear=True,
        n_flip=1,
        sim_id=idd,
        real_data=True,
    )
    sim_dict[idd] = config


for linear in [True, False]:
    lin_id = "linear" if linear else "nonlinear"
    for sample_size_train in [400, 700, 1400, 2000]:
        sim_id = "sample_size_train_{}_{}".format(sample_size_train, lin_id)
        config = DataGeneratorConfig(
            n_confounder=N_CONFOUNDER,
            n_cause=N_CAUSE,
            n_outcome=1,
            sample_size=SAMPLE_SIZE,
            p_confounder_cause=P_CONFOUNDER_CAUSE,
            confounding_level=CONFOUNDING_LEVEL,
            p_cause_cause=P_CAUSE_CAUSE,
            cause_noise=CAUSE_NOISE,
            outcome_noise=OUTCOME_NOISE,
            linear=linear,
            n_flip=1,
            sim_id=sim_id,
            sample_size_train=sample_size_train,
            outcome_interaction=OUTCOME_INTERACTION,
        )
        sim_dict[sim_id] = config

    for n_confounder in [10, 20, 30, 40]:
        sim_id = "n_confounder_{}_{}".format(n_confounder, lin_id)
        config = DataGeneratorConfig(
            n_confounder=n_confounder,
            n_cause=N_CAUSE,
            n_outcome=1,
            sample_size=SAMPLE_SIZE,
            p_confounder_cause=P_CONFOUNDER_CAUSE,
            confounding_level=CONFOUNDING_LEVEL,
            p_cause_cause=P_CAUSE_CAUSE,
            cause_noise=CAUSE_NOISE,
            outcome_noise=OUTCOME_NOISE,
            linear=linear,
            n_flip=1,
            sim_id=sim_id,
            outcome_interaction=OUTCOME_INTERACTION,
        )
        sim_dict[sim_id] = config

    for confounding_level in [1.0, 3.0, 5.0, 7.0]:
        sim_id = "confounding_level_{}_{}".format(confounding_level, lin_id)
        config = DataGeneratorConfig(
            n_confounder=N_CONFOUNDER,
            n_cause=N_CAUSE,
            n_outcome=1,
            sample_size=SAMPLE_SIZE,
            p_confounder_cause=P_CONFOUNDER_CAUSE,
            p_cause_cause=P_CAUSE_CAUSE,
            cause_noise=CAUSE_NOISE,
            outcome_noise=OUTCOME_NOISE,
            linear=linear,
            n_flip=1,
            confounding_level=confounding_level,
            sim_id=sim_id,
            outcome_interaction=OUTCOME_INTERACTION,
        )
        sim_dict[sim_id] = config

    for n_flip in [1, 2, 3, 5]:
        sim_id = "n_flip_{}_{}".format(n_flip, lin_id)
        config = DataGeneratorConfig(
            n_confounder=N_CONFOUNDER,
            n_cause=N_CAUSE,
            n_outcome=1,
            confounding_level=CONFOUNDING_LEVEL,
            sample_size=SAMPLE_SIZE,
            p_confounder_cause=P_CONFOUNDER_CAUSE,
            p_cause_cause=P_CAUSE_CAUSE,
            cause_noise=CAUSE_NOISE,
            outcome_noise=OUTCOME_NOISE,
            linear=linear,
            n_flip=n_flip,
            sim_id=sim_id,
            outcome_interaction=OUTCOME_INTERACTION,
        )
        sim_dict[sim_id] = config

    for p_confounder_cause in [0.1, 0.3, 0.5, 0.8]:
        sim_id = "p_confounder_cause_{}_{}".format(p_confounder_cause, lin_id)
        config = DataGeneratorConfig(
            n_confounder=N_CONFOUNDER,
            n_cause=N_CAUSE,
            n_outcome=1,
            confounding_level=CONFOUNDING_LEVEL,
            sample_size=SAMPLE_SIZE,
            p_confounder_cause=p_confounder_cause,
            p_cause_cause=P_CAUSE_CAUSE,
            cause_noise=CAUSE_NOISE,
            outcome_noise=OUTCOME_NOISE,
            linear=linear,
            sim_id=sim_id,
            outcome_interaction=OUTCOME_INTERACTION,
        )
        sim_dict[sim_id] = config

    for p_cause_cause in [0.1, 0.3, 0.5, 0.8]:
        sim_id = "p_cause_cause_{}_{}".format(p_cause_cause, lin_id)
        config = DataGeneratorConfig(
            n_confounder=N_CONFOUNDER,
            n_cause=N_CAUSE,
            n_outcome=1,
            confounding_level=CONFOUNDING_LEVEL,
            sample_size=SAMPLE_SIZE,
            p_confounder_cause=P_CONFOUNDER_CAUSE,
            p_cause_cause=p_cause_cause,
            cause_noise=CAUSE_NOISE,
            outcome_noise=OUTCOME_NOISE,
            linear=linear,
            sim_id=sim_id,
            n_flip=1,
            outcome_interaction=OUTCOME_INTERACTION,
        )
        sim_dict[sim_id] = config

    for p_outcome_single in [0.1, 0.3, 0.5, 0.8]:
        sim_id = "p_outcome_single_{}_{}".format(p_outcome_single, lin_id)
        config = DataGeneratorConfig(
            n_confounder=N_CONFOUNDER,
            n_cause=N_CAUSE,
            n_outcome=1,
            confounding_level=CONFOUNDING_LEVEL,
            sample_size=SAMPLE_SIZE,
            p_confounder_cause=P_CONFOUNDER_CAUSE,
            p_cause_cause=P_CAUSE_CAUSE,
            p_outcome_single=p_outcome_single,
            p_outcome_double=P_OUTCOME_DOUBLE,
            cause_noise=CAUSE_NOISE,
            outcome_noise=OUTCOME_NOISE,
            linear=linear,
            sim_id=sim_id,
            n_flip=1,
            outcome_interaction=OUTCOME_INTERACTION,
        )
        sim_dict[sim_id] = config

    for cause_noise in [0.01, 0.3, 0.5, 0.8]:
        sim_id = "cause_noise_{}_{}".format(cause_noise, lin_id)
        config = DataGeneratorConfig(
            n_confounder=N_CONFOUNDER,
            n_cause=N_CAUSE,
            n_outcome=1,
            confounding_level=CONFOUNDING_LEVEL,
            sample_size=SAMPLE_SIZE,
            p_confounder_cause=P_CONFOUNDER_CAUSE,
            p_cause_cause=P_CAUSE_CAUSE,
            p_outcome_single=P_OUTCOME_SINGLE,
            p_outcome_double=P_OUTCOME_DOUBLE,
            cause_noise=cause_noise,
            outcome_noise=OUTCOME_NOISE,
            linear=linear,
            sim_id=sim_id,
            n_flip=1,
            outcome_interaction=OUTCOME_INTERACTION,
        )
        sim_dict[sim_id] = config

    for p_outcome_double in [0.05, 0.1, 0.15, 0.2]:
        sim_id = "p_outcome_double_{}_{}".format(p_outcome_double, lin_id)
        config = DataGeneratorConfig(
            n_confounder=N_CONFOUNDER,
            n_cause=N_CAUSE,
            n_outcome=1,
            confounding_level=CONFOUNDING_LEVEL,
            sample_size=SAMPLE_SIZE,
            p_confounder_cause=P_CONFOUNDER_CAUSE,
            p_cause_cause=P_CAUSE_CAUSE,
            p_outcome_single=P_OUTCOME_SINGLE,
            p_outcome_double=p_outcome_double,
            cause_noise=CAUSE_NOISE,
            outcome_noise=OUTCOME_NOISE,
            linear=linear,
            sim_id=sim_id,
            n_flip=1,
            outcome_interaction=OUTCOME_INTERACTION,
        )
        sim_dict[sim_id] = config

    for n_cause in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        sim_id = "n_cause_{}_{}".format(n_cause, lin_id)
        config = DataGeneratorConfig(
            n_confounder=N_CONFOUNDER,
            n_cause=n_cause,
            n_outcome=1,
            confounding_level=CONFOUNDING_LEVEL,
            sample_size=SAMPLE_SIZE,
            p_confounder_cause=P_CONFOUNDER_CAUSE,
            p_cause_cause=P_CAUSE_CAUSE,
            cause_noise=CAUSE_NOISE,
            outcome_noise=OUTCOME_NOISE,
            linear=linear,
            sim_id=sim_id,
            outcome_interaction=OUTCOME_INTERACTION,
        )
        sim_dict[sim_id] = config

    for n_cause in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        sim_id = "nc_n_cause_{}_{}".format(n_cause, lin_id)
        config = DataGeneratorConfig(
            n_confounder=2,
            n_cause=n_cause,
            n_outcome=1,
            confounding_level=CONFOUNDING_LEVEL,
            sample_size=SAMPLE_SIZE,
            p_confounder_cause=0.0,
            p_cause_cause=0.0,
            cause_noise=CAUSE_NOISE,
            outcome_noise=OUTCOME_NOISE,
            linear=linear,
            sim_id=sim_id,
            outcome_interaction=OUTCOME_INTERACTION,
            p_outcome_single=1.0,
            p_outcome_double=1.0,
        )
        sim_dict[sim_id] = config

    for n_cause in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        sim_id = "ndlm_n_cause_{}_{}".format(n_cause, lin_id)
        config = DataGeneratorConfig(
            n_confounder=N_CONFOUNDER,
            n_cause=n_cause,
            n_outcome=1,
            confounding_level=10,
            sample_size=SAMPLE_SIZE,
            p_confounder_cause=0.5,
            p_cause_cause=0.0,
            cause_noise=0.0,
            outcome_noise=0.1,
            linear=linear,
            sim_id=sim_id,
            outcome_interaction=OUTCOME_INTERACTION,
            p_outcome_single=0.2,
            p_outcome_double=0.001,
        )
        sim_dict[sim_id] = config

    for n_cause in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        sim_id = "ldlm_n_cause_{}_{}".format(n_cause, lin_id)
        config = DataGeneratorConfig(
            n_confounder=N_CONFOUNDER,
            n_cause=n_cause,
            n_outcome=1,
            confounding_level=CONFOUNDING_LEVEL,
            sample_size=SAMPLE_SIZE,
            p_confounder_cause=0.5,
            p_cause_cause=0.0,
            cause_noise=0.0,
            outcome_noise=0.1,
            linear=linear,
            sim_id=sim_id,
            outcome_interaction=OUTCOME_INTERACTION,
            p_outcome_single=1.0,
            p_outcome_double=0.0,
        )
        sim_dict[sim_id] = config

        for n_cause in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            sim_id = "o3_n_cause_{}_{}".format(n_cause, lin_id)
            config = DataGeneratorConfig(
                n_confounder=5,
                n_cause=n_cause,
                n_outcome=1,
                confounding_level=CONFOUNDING_LEVEL,
                sample_size=SAMPLE_SIZE,
                p_confounder_cause=0.5,
                p_cause_cause=P_CAUSE_CAUSE,
                cause_noise=CAUSE_NOISE,
                outcome_noise=0.1,
                linear=linear,
                sim_id=sim_id,
                outcome_interaction=OUTCOME_INTERACTION,
                p_outcome_single=1.0,
                p_outcome_double=1.0,
            )
            sim_dict[sim_id] = config

        for n_cause in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            sim_id = "dn_n_cause_{}_{}".format(n_cause, lin_id)
            config = DataGeneratorConfig(
                n_confounder=5,
                n_cause=n_cause,
                n_outcome=1,
                confounding_level=CONFOUNDING_LEVEL,
                sample_size=SAMPLE_SIZE,
                p_confounder_cause=1.0,
                p_cause_cause=1.0,
                cause_noise=CAUSE_NOISE,
                outcome_noise=0.1,
                linear=linear,
                sim_id=sim_id,
                outcome_interaction=OUTCOME_INTERACTION,
                p_outcome_single=1.0,
                p_outcome_double=1.0,
            )
            sim_dict[sim_id] = config

for n_flip in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    sim_id = "abl_n_flip_{}".format(n_flip)
    config = DataGeneratorConfig(
        n_confounder=N_CONFOUNDER,
        n_cause=10,
        n_outcome=1,
        sample_size=5000,
        p_confounder_cause=0.3,
        p_cause_cause=0.9,
        cause_noise=0.0,
        outcome_noise=0.01,
        n_flip=n_flip,
        confounding_level=1.0,
        linear=True,
        sim_id=sim_id,
    )
    sim_dict[sim_id] = config

for n_flip in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    sim_id = "abl_n_flip_new_{}".format(n_flip)
    config = DataGeneratorConfig(
        n_confounder=N_CONFOUNDER,
        n_cause=10,
        n_outcome=1,
        sample_size=5000,
        p_confounder_cause=P_CONFOUNDER_CAUSE,
        p_cause_cause=0.9,
        cause_noise=CAUSE_NOISE,
        outcome_noise=0.01,
        n_flip=n_flip,
        confounding_level=CONFOUNDING_LEVEL,
        linear=True,
        sim_id=sim_id,
        outcome_interaction=OUTCOME_INTERACTION,
        p_outcome_single=P_OUTCOME_SINGLE,
        p_outcome_double=P_OUTCOME_DOUBLE,
    )
    sim_dict[sim_id] = config


for p_cause_cause in [0.1, 0.3, 0.5, 0.8]:
    sim_id = "abl_p_cause_cause_{}".format(p_cause_cause)
    config = DataGeneratorConfig(
        n_confounder=N_CONFOUNDER,
        n_cause=5,
        n_outcome=1,
        sample_size=5000,
        p_confounder_cause=0.3,
        p_cause_cause=p_cause_cause,
        cause_noise=0.01,
        outcome_noise=0.01,
        linear=True,
        confounding_level=1.0,
        sim_id=sim_id,
        n_flip=2,
    )
    sim_dict[sim_id] = config


for p_confounder_cause in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    sim_id = "abl_p_confounder_cause_{}".format(p_confounder_cause)
    config = DataGeneratorConfig(
        n_confounder=N_CONFOUNDER,
        n_cause=10,
        n_outcome=1,
        sample_size=5000,
        p_confounder_cause=p_confounder_cause,
        p_cause_cause=0.0,
        cause_noise=0.0,
        outcome_noise=0.01,
        linear=True,
        confounding_level=1.0,
        sim_id=sim_id,
        n_flip=1,
    )
    sim_dict[sim_id] = config

sim_id = "ea_{}_{}".format(5, "linear")
config = DataGeneratorConfig(
    n_confounder=N_CONFOUNDER,
    n_cause=5,
    n_outcome=1,
    confounding_level=CONFOUNDING_LEVEL,
    sample_size=SAMPLE_SIZE,
    p_confounder_cause=P_CONFOUNDER_CAUSE,
    p_cause_cause=P_CAUSE_CAUSE,
    cause_noise=CAUSE_NOISE,
    outcome_noise=0.1,
    linear=True,
    sim_id=sim_id,
    outcome_interaction=OUTCOME_INTERACTION,
    n_flip=1,
)
sim_dict[sim_id] = config

for confounding_level in [2, 4, 6, 8, 10]:
    sim_id = "ea_balance_{}".format(confounding_level)
    config = DataGeneratorConfig(
        n_confounder=5,
        n_cause=5,
        n_outcome=1,
        confounding_level=confounding_level,
        sample_size=SAMPLE_SIZE,
        p_confounder_cause=1,
        p_cause_cause=1,
        cause_noise=0,
        outcome_noise=0.1,
        linear=True,
        sim_id=sim_id,
        outcome_interaction=OUTCOME_INTERACTION,
        n_flip=1,
    )
    sim_dict[sim_id] = config


for flip in range(0, 6):
    sim_id = "ea_{}_{}_{}".format(5, "linear", flip)
    config = DataGeneratorConfig(
        n_confounder=N_CONFOUNDER,
        n_cause=5,
        n_outcome=1,
        confounding_level=CONFOUNDING_LEVEL,
        sample_size=SAMPLE_SIZE,
        p_confounder_cause=P_CONFOUNDER_CAUSE,
        p_cause_cause=P_CAUSE_CAUSE,
        cause_noise=CAUSE_NOISE,
        outcome_noise=0.1,
        linear=True,
        sim_id=sim_id,
        outcome_interaction=OUTCOME_INTERACTION,
        n_flip=flip,
    )
    sim_dict[sim_id] = config


# p_confounder_cause=0.1, 0.3, 0.5
# independent_cause = DataGeneratorConfig(
#     n_confounder=N_CONFOUNDER,
#     n_cause=N_CAUSE,
#     n_outcome=1,
#     sample_size=SAMPLE_SIZE,
#     # higher, higher error
#     p_confounder_cause=P_CONFOUNDER_CAUSE,
#     # independent cause
#     p_cause_cause=0.0,
#     cause_noise=CAUSE_NOISE,
#     outcome_noise=OUTCOME_NOISE,
#     linear=True
# )

# n_cause = 2, 5, 10
# independent_cause = DataGeneratorConfig(
#     n_confounder=N_CONFOUNDER,
#     n_cause=N_CAUSE,
#     n_outcome=1,
#     sample_size=SAMPLE_SIZE,
#     # higher, higher error
#     p_confounder_cause=P_CONFOUNDER_CAUSE,
#     # independent cause
#     p_cause_cause=0.0,
#     cause_noise=CAUSE_NOISE,
#     outcome_noise=OUTCOME_NOISE,
#     linear=True
# )


class AblationConfig(NamedTuple):
    perturb_subset: int = 10000
    exact_single_cause: bool = False
    predict_all_causes: bool = False
    is_ablate: bool = False
    ablation_id: str = ""
    oracle_po: bool = False
    oracle_potential_cause: bool = False
