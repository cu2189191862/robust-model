from supplychainnetworkmodel import SupplyChainNetworkModel
from schedulingmodel import SchedulingModel
from typing import Dict

import os
import numpy as np
import pandas as pd
import sys


def scheduling_examples(robustness=1) -> None:
    sample_count = 30
    log_dir = os.path.join("logs", "sch-models")
    figure_dir = os.path.join("plots", "sch-models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"sch-example.txt")
    schm = SchedulingModel(name="scheduling-model",
                           log_path=log_path, figure_dir=figure_dir)
    schm.init_sets(T_count=30)
    schm.init_parameters()
    schm.init_uncertainties(robustness=robustness)
    schm.init_variables()
    schm.init_objective()
    schm.init_constraints()
    schm.log_origin_model()
    schm.transform(verbose=False)
    schm.log_robust_counterpart()
    schm.solve()
    schm.log_solution()
    mvor, iostd, rr = schm.evaluate(sample=sample_count, seed=5, verbose=False)
    # ro_sols = schm.get_robust_solutions()
    # det_sols = schm.get_deterministic_solutions()
    # for t in schm.T:
    #     print(ro_sols[schm.s[t]], det_sols[schm.s[t]])
    return mvor, iostd, rr


def scn_examples(robustness=0.3) -> Dict[str, float]:
    sample_count = 30
    log_dir = os.path.join("logs", "scn-models")
    figure_dir = os.path.join("plots", "scn-models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    objectives = list()

    replication_count = 1
    for i in range(replication_count):
        log_path = os.path.join(log_dir, f"scn-example-{i}.txt")
        figure_dir = os.path.join(figure_dir)
        scnm = SupplyChainNetworkModel(
            name="supply-chain-network-model", log_path=log_path, figure_dir=figure_dir)
        scnm.init_sets()
        # scnm.init_sets(I_count=10, J_count=10, M_count=10,
        #                N_count=4, K_count=15, L_count=15)
        scnm.init_parameters(seed=7)  # fixed for each replication
        # sampled for each replication
        scnm.init_uncertainties(
            seed=i, robustness=robustness, using_data=False)
        scnm.init_variables()
        scnm.init_objective()
        scnm.init_constraints()
        scnm.log_origin_model()
        scnm.transform(verbose=False)
        scnm.log_robust_counterpart()
        scnm.solve()
        scnm.log_solution()
        objectives.append(scnm.get_robust_objective_value())
        # evaluation
        mvor, iostd, rr = scnm.evaluate(
            sample=sample_count, seed=5, verbose=False)
        print(f"replication-{i} model expected objective: Robust = {scnm.get_robust_objective_value()}, Determined (lower bound) = {scnm.get_deterministic_model_objective_value()}")
        # global ro_sols
        # global det_sols
        # ro_sols = scnm.get_robust_solutions()
        # det_sols = scnm.get_deterministic_solutions()

    # print(f"mean: {np.mean(objectives)}, std: {np.std(objectives)}")
    return mvor, iostd, rr


def main() -> None:
    # mvors = list()
    # iostds = list()
    # rrs = list()
    # for r in np.arange(0.3, 0.5, 0.01):
    #     print(f"-------------------{r}------------------")
    #     mvor, iostd, rr = scn_examples(robustness=r)
    #     mvors.append(mvor)
    #     iostds.append(iostd)
    #     rrs.append(rr)
    #     print(mvor, iostd, rr)
    # records = pd.DataFrame(zip(mvors, iostds, rrs), columns=[
    #                        "mvors", "iostds", "rrs"])
    # records.to_csv("records.csv", index=False)

    # print("mean value of robustness (PM)", scheduling_examples(robustness=1))
    # scheduling_examples()


    model_type = sys.argv[1]
    if model_type == "--scn":
        mvor, iostd, rr = scn_examples()
    elif model_type == "--sch":
        mvor, iostd, rr = scheduling_examples()
    else:
        raise ValueError(f"Invalid model type argv: {model_type}")
    print(f'mean_value_of_robustization: {mvor}, improvement_of_std: {iostd}, robust_rate: {rr}')

if __name__ == "__main__":
    main()
