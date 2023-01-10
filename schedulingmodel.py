from robustoptimization.robustlinearmodel import RobustLinearModel
from robustoptimization.components.uncertainparameter import UncertainParameter
from robustoptimization.components.uncertaintyset.box import Box
from robustoptimization.utils.constants import *
from robustoptimization.utils.plotter import generate_evaluations_plot
from robustoptimization.utils.metrics import improvement_of_std, mean_value_of_robustization, robust_rate
from datetime import datetime
from typing import Union
from tqdm import tqdm
import random
import os


class SchedulingModel():
    def __init__(self, name: str, log_path: str, figure_dir: str) -> None:
        self.start_timestamp = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
        self.rlm = RobustLinearModel(log_path, name, GUROBI)
        # self.data_point_count = 5
        self.figure_dir = figure_dir

    def init_sets(self, T_count: int = 20) -> None:
        self.T = set(range(T_count))

    def init_parameters(self, seed: int = 5) -> None:
        random.seed(seed)
        self.C_S = 700
        self.C_H = 5
        self.C_CM = 100
        self.C_PM = 500
        self.T_PM = 1
        self.T_CM = 2
        self.H_max = 0
        self.H_I = 0
        self.B_I = 10
        self.S_I = 0.97
        self.R = 0.25
        self.Z = 0.6
        self.M = 1000000000
        self.D_star = UncertainParameter("D_star")
        self.Q = [round(random.uniform(100, 150), 2) for _ in self.T]
        self.Y = 10000

    def init_uncertainties(self, robustness: float) -> None:
        nominal_data = {self.D_star: 0.8}
        base_shifts = [{self.D_star: 0.1}]

        self.deterior_unc_set = Box(
            "deterior_box", robustness=robustness, nominal_data=nominal_data, base_shifts=base_shifts)

    def init_variables(self) -> None:
        self.x = dict()
        self.z_PM = dict()
        self.z_CM = dict()
        self.s = dict()
        self.z_P = dict()
        self.b = dict()
        self.h = dict()
        for t in self.T:
            self.x[t] = self.rlm.add_var(f'x_{t}', lb=0, type_=CONTINUOUS)
            self.z_PM[t] = self.rlm.add_var(f'z_PM_{t}', type_=BINARY)
            self.z_CM[t] = self.rlm.add_var(f'z_CM_{t}', type_=BINARY)
            self.s[t] = self.rlm.add_var(
                f's_{t}', lb=0, ub=1, type_=CONTINUOUS)
            self.z_P[t] = self.rlm.add_var(f'z_P_{t}', type_=BINARY)
            self.b[t] = self.rlm.add_var(f'b_{t}', lb=0, type_=CONTINUOUS)
            self.h[t] = self.rlm.add_var(f'h_{t}', lb=0, type_=CONTINUOUS)

    def init_objective(self) -> None:
        h_sumover_t = 0
        b_sumover_t = 0
        z_PM_sumover_t = 0
        z_CM_sumover_t = 0
        for t in self.T:
            h_sumover_t += self.h[t]
            b_sumover_t += self.b[t]
            z_PM_sumover_t += self.z_PM[t]
            z_CM_sumover_t += self.z_CM[t]
        objective = self.C_H * h_sumover_t + self.C_S * b_sumover_t + \
            self.C_PM * z_PM_sumover_t + self.C_CM * z_CM_sumover_t
        self.rlm.set_objective(objective, sense=MINIMIZE)
        # self.rlm.set_objective(-objective, sense=MAXIMIZE)

    def init_constraints(self) -> None:
        self.__c1()  # States initialization.
        self.__c2()  # Exclusivity of PM, CM, and production.
        # Machine can only be resting, starting PM, starting CM, producing, or maintaining.
        self.__c3()
        # Adopt CM if the health status of the machine less than the threshold.
        self.__c4()
        # Health status of the machine recover after starting PM or CM.
        self.__c5()
        # Health status of the machine deteriorate in a rate if is setup.
        self.__c6()
        self.__c7()  # Determine setup or not.
        # If not producing or adopting PM / CM, the machine stage should stay unchanged.
        self.__c8()
        self.__c9()  # Determine the largest yield amount.
        # Maintainence of holding, stock out, and supply / demand balance.
        self.__c10()
        self.__c11()  # Maximum inventory level.

    def log_origin_model(self) -> None:
        self.rlm.log_origin_model()

    def log_robust_counterpart(self) -> None:
        self.rlm.log_robust_counterpart()

    def transform(self, verbose: bool = False) -> None:
        self.rlm.transform(verbose=verbose)

    def solve(self) -> None:
        self.rlm.solve()

    def log_solution(self) -> None:
        self.rlm.log_solution()

    def get_robust_solutions(self):
        return self.rlm.get_robust_solutions()

    def evaluate(self, sample: int = 10, seed=5, verbose: bool = True):
        random.seed(seed)
        # make deterministic model first, and then compare performances.
        self.rlm.make_deterministic_model_using_nominal_data()
        # TODO: just pseudo code
        decisions = list(self.z_PM.values())
        comparisons = {
            "deterministic": list(),
            "robust": list()
        }
        wondered = [self.b[t] for t in self.T]
        for _ in tqdm(range(sample)):
            deter_real, robust_real, deter_wondered, robust_wondered = self.rlm.evaluator(realization=self.__realize(),
                                                                                          decisions=decisions, wondered=wondered)
            print(sum(deter_wondered.values()))
            print(sum(robust_wondered.values()))
            comparisons["deterministic"].append(deter_real)
            comparisons["robust"].append(robust_real)
            if deter_real == None:
                deter_real = "infeasible"
            if robust_real == None:
                robust_real = "infeasible"
            if verbose:
                print(f"scenario {_}")
                print("deterministic realization:", deter_real)
                print("robust realization:", robust_real)
        generate_evaluations_plot(figure_path=os.path.join(self.figure_dir, "evaluations.png"),
                                  comparisons=comparisons, title=f"{self.rlm.name} evaluations plot")
        return mean_value_of_robustization(comparisons, self.rlm.sense), improvement_of_std(comparisons, self.rlm.sense), robust_rate(comparisons)

    def get_deterministic_solutions(self):
        return self.rlm.get_deterministic_solutions()

    def get_robust_objective_value(self) -> Union[float, int]:
        return self.rlm.get_robust_objective_value()

    def get_deterministic_model_objective_value(self) -> Union[float, int]:
        return self.rlm.get_deterministic_model_objective_value()

    def __realize(self):
        realization = dict()
        # realization[self.D_star] = random.normalvariate(0.8, 0.05)
        realization[self.D_star] = random.uniform(0.7, 0.9)
        return realization

    def __c1(self):
        self.rlm.add_constraint(
            self.x[0] <= 0
        )
        self.rlm.add_constraint(
            self.z_PM[0] <= 0
        )
        self.rlm.add_constraint(
            self.z_CM[0] <= 0
        )
        self.rlm.add_constraint(
            self.z_P[0] <= 0
        )
        self.rlm.add_constraint(
            self.h[0] <= self.H_I
        )
        self.rlm.add_constraint(
            self.b[0] <= self.B_I
        )
        self.rlm.add_constraint(
            self.s[0] <= self.S_I
        )

    def __c2(self):
        for t in self.T:
            z_PM_CM_sumover_t_prime = 0
            for t_prime in self.T:
                if t_prime >= t + 1 and t_prime <= t + self.T_PM - 1:
                    z_PM_CM_sumover_t_prime += self.z_PM[t_prime] +\
                        self.z_CM[t_prime]
            self.rlm.add_constraint(
                (1 - self.z_PM[t]) * self.M >= z_PM_CM_sumover_t_prime
            )
            z_PM_CM_sumover_t_prime = 0
            for t_prime in self.T:
                if t_prime >= t + 1 and t_prime <= t + self.T_CM - 1:
                    z_PM_CM_sumover_t_prime += self.z_PM[t_prime] +\
                        self.z_CM[t_prime]
            self.rlm.add_constraint(
                (1 - self.z_CM[t]) * self.M >= z_PM_CM_sumover_t_prime
            )

            x_sumover_t_prime = 0
            for t_prime in self.T:
                if t_prime >= t and t_prime <= t + self.T_PM - 1:
                    x_sumover_t_prime += self.x[t_prime]
            self.rlm.add_constraint(
                (1 - self.z_PM[t]) * self.M >= x_sumover_t_prime
            )

            x_sumover_t_prime = 0
            for t_prime in self.T:
                if t_prime >= t and t_prime <= t + self.T_CM - 1:
                    x_sumover_t_prime += self.x[t_prime]
            self.rlm.add_constraint(
                (1 - self.z_CM[t]) * self.M >= x_sumover_t_prime
            )

    def __c3(self):
        for t in self.T:
            self.rlm.add_constraint(
                self.z_PM[t] + self.z_CM[t] + self.z_P[t] <= 1
            )

    def __c4(self):
        for t in self.T:
            self.rlm.add_constraint(
                self.z_CM[t] >= (self.Z - self.s[t])
            )

    def __c5(self):
        for t in self.T:
            if t != len(self.T) - 1:
                self.rlm.add_constraint(
                    self.z_CM[t] <= self.s[t+1]
                )
                self.rlm.add_constraint(
                    self.s[t+1] <= (self.s[t] + self.R) +
                    self.M * (1 - self.z_PM[t])
                )

    def __c6(self):
        for t in self.T:
            if t != len(self.T) - 1:
                self.rlm.add_constraint(
                    self.s[t+1] <= self.D_star *
                    self.s[t] + self.M * (1 - self.z_P[t]), uncertainty_set=self.deterior_unc_set
                )

    def __c7(self):
        for t in self.T:
            self.rlm.add_constraint(
                self.M * self.z_P[t] >= self.x[t]
            )
            self.rlm.add_constraint(
                self.x[t] >= self.z_P[t]
            )

    def __c8(self):
        for t in self.T:
            if t != len(self.T) - 1:
                self.rlm.add_constraint(
                    self.s[t+1] >= self.s[t] - self.M *
                    (self.z_P[t] + self.z_PM[t] + self.z_CM[t])
                )
                self.rlm.add_constraint(
                    self.s[t+1] <= self.s[t] + self.M *
                    (self.z_P[t] + self.z_PM[t] + self.z_CM[t])
                )

    def __c9(self):
        for t in self.T:
            self.rlm.add_constraint(
                self.x[t] <= self.Y * self.s[t]
            )

    def __c10(self):
        for t in self.T:
            if t != 0:
                self.rlm.add_constraint(
                    self.h[t] <= self.x[t] + self.h[t-1] -
                    (self.Q[t] + self.b[t-1]) + self.b[t]
                )

    def __c11(self):
        for t in self.T:
            self.rlm.add_constraint(
                self.h[t] <= self.H_max
            )
