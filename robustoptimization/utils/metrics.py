from typing import Dict, Union, List

from robustoptimization.utils.constants import *
import numpy as np


def exception_handler(comparisons: Dict[str, float], sense: Union[MINIMIZE, MAXIMIZE] = MINIMIZE):
    if "robust" not in comparisons or "deterministic" not in comparisons:
        raise ValueError(
            "Essential key 'robust' or 'deterministic' not find in comparisons dictionary.")
    if len(comparisons["robust"]) != len(comparisons["deterministic"]):
        raise ValueError("Length of the comparison lists inconsist.")
    if sense != MINIMIZE and sense != MAXIMIZE:
        raise ValueError(f"Invalid sense '{sense}'.")


def mean_value_of_robustization(comparisons: Dict[str, List[float]], sense: Union[MINIMIZE, MAXIMIZE] = MINIMIZE) -> float:
    '''Metric evaluates the value of robustization
    '''
    exception_handler(comparisons, sense)

    observation_count = len(comparisons["robust"])
    value = 0
    if sense == MINIMIZE:
        robust_worst = max([i for i in comparisons["robust"] if i is not None])
        deterministic_worst = max(
            [i for i in comparisons["deterministic"] if i is not None])
        worst = max(robust_worst, deterministic_worst)
        for i in range(observation_count):
            if comparisons["deterministic"][i] != None and comparisons["robust"][i] != None:
                value += comparisons["deterministic"][i] - \
                    comparisons["robust"][i]
            elif comparisons["deterministic"][i] != None and comparisons["robust"][i] == None:
                value += comparisons["deterministic"][i] - worst
            elif comparisons["deterministic"][i] == None and comparisons["robust"][i] != None:
                value += worst - comparisons["robust"][i]
            elif comparisons["deterministic"][i] == None and comparisons["robust"][i] == None:
                pass

        return value / observation_count
    else:
        robust_worst = min([i for i in comparisons["robust"] if i is not None])
        deterministic_worst = min(
            [i for i in comparisons["deterministic"] if i is not None])
        worst = min(robust_worst, deterministic_worst)
        for i in range(observation_count):
            if comparisons["deterministic"][i] != None and comparisons["robust"][i] != None:
                value += comparisons["robust"][i] - \
                    comparisons["deterministic"][i]
            elif comparisons["deterministic"][i] != None and comparisons["robust"][i] == None:
                value += worst - comparisons["deterministic"][i]
            elif comparisons["deterministic"][i] == None and comparisons["robust"][i] != None:
                value += comparisons["robust"][i] - worst
            elif comparisons["deterministic"][i] == None and comparisons["robust"][i] == None:
                pass

        return value / observation_count


def improvement_of_std(comparisons: Dict[str, List[float]], sense: Union[MINIMIZE, MAXIMIZE] = MINIMIZE) -> float:
    exception_handler(comparisons, sense)
    if sense == MINIMIZE:
        robust_worst = max([i for i in comparisons["robust"] if i is not None])
        deterministic_worst = max(
            [i for i in comparisons["deterministic"] if i is not None])
        worst = max(robust_worst, deterministic_worst)
        r = [i if i != None else worst for i in comparisons["robust"]]
        d = [i if i != None else worst for i in comparisons["deterministic"]]

        return np.std(d) / np.std(r)
    else:
        robust_worst = min([i for i in comparisons["robust"] if i is not None])
        deterministic_worst = min(
            [i for i in comparisons["deterministic"] if i is not None])
        worst = min(robust_worst, deterministic_worst)
        r = [i if i != None else worst for i in comparisons["robust"]]
        d = [i if i != None else worst for i in comparisons["deterministic"]]

        return np.std(d) / np.std(r)


def robust_rate(comparisons: Dict[str, List[float]],) -> float:
    exception_handler(comparisons)
    robust_feasible_deterministic_infeasible = 0
    robust_feasible = 0
    for i in range(len(comparisons["robust"])):
        if comparisons["robust"][i] != None and comparisons["deterministic"][i] == None:
            robust_feasible_deterministic_infeasible += 1
        if comparisons["robust"][i] != None:
            robust_feasible += 1

    return robust_feasible_deterministic_infeasible / robust_feasible
