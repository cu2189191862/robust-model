from robustoptimization.robustlinearmodel import RobustLinearModel
from robustoptimization.components.uncertainparameter import UncertainParameter
from robustoptimization.components.uncertaintyset.box import Box
from robustoptimization.utils.constants import *
from robustoptimization.utils.plotter import generate_evaluations_plot
from robustoptimization.utils.metrics import mean_value_of_robustization, improvement_of_std, robust_rate
from datetime import datetime
from typing import Union
from tqdm import tqdm

import os
import random


class SupplyChainNetworkModel():
    def __init__(self, name: str, log_path: str, figure_dir: str) -> None:
        self.start_timestamp = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
        self.rlm = RobustLinearModel(log_path, name)
        self.data_point_count = 5
        self.figure_dir = figure_dir

    def init_sets(self, I_count: int = 5, J_count: int = 3, M_count: int = 5, N_count: int = 2, K_count: int = 10, L_count: int = 10) -> None:
        self.I = set(range(I_count))
        self.J = set(range(J_count))
        self.M = set(range(M_count))
        self.N = set(range(N_count))
        self.K = set(range(K_count))
        self.L = set(range(L_count))

    def init_parameters(self, seed: int = 5) -> None:
        random.seed(seed)
        self.D_star = [UncertainParameter(f"D_star_{l}") for l in self.L]
        self.R_star = [UncertainParameter(f"R_star_{k}") for k in self.K]
        self.S = 0.2
        self.C_C = [round(random.uniform(1500, 2000), 2) for i in self.I]
        self.C_R = [round(random.uniform(2000, 3000), 2) for j in self.J]
        self.C_E = [round(random.uniform(1500, 2000), 2) for m in self.I]
        self.C_D = [round(random.uniform(800, 1000), 2) for n in self.I]
        self.F = [round(random.uniform(210000, 2400000), 2) for i in self.I]
        self.G = [round(random.uniform(4500000, 4900000), 2) for j in self.J]
        self.H = [round(random.uniform(160000, 200000), 2) for m in self.M]
        self.C_star = [
            [UncertainParameter(f"C_star_{k}_{i}") for i in self.I] for k in self.K]
        self.A_star = [
            [UncertainParameter(f"A_star_{i}_{j}") for j in self.J] for i in self.I]
        self.B_star = [
            [UncertainParameter(f"B_star_{j}_{m}") for m in self.M] for j in self.J]
        self.E_star = [
            [UncertainParameter(f"E_star_{m}_{l}") for l in self.L] for m in self.M]
        self.V_star = [
            [UncertainParameter(f"V_star_{i}_{n}") for n in self.N] for i in self.I]
        self.P = [round(random.uniform(4500, 6000), 2) for l in self.L]

    def init_uncertainties(self, seed: int, robustness: float, using_data: bool = True) -> None:
        if using_data:
            random.seed(seed)
            self.__data_objective_uncertainty_set(robustness)
            self.__data_demand_uncertainty_sets(robustness)
            self.__data_return_uncertainty_sets(robustness)
        else:
            self.__range_objective_uncertainty_set(robustness)
            self.__range_demand_uncertainty_sets(robustness)
            self.__range_return_uncertainty_sets(robustness)

    def init_variables(self) -> None:
        self.x = dict()
        for k in self.K:
            for i in self.I:
                self.x[k, i] = self.rlm.add_var(f'x_{k}_{i}')

        self.u = dict()
        for i in self.I:
            for j in self.J:
                self.u[i, j] = self.rlm.add_var(f'u_{i}_{j}')

        self.p = dict()
        for j in self.J:
            for m in self.M:
                self.p[j, m] = self.rlm.add_var(f'p_{j}_{m}')

        self.q = dict()
        for m in self.M:
            for l in self.L:
                self.q[m, l] = self.rlm.add_var(f'q_{m}_{l}')

        self.t = dict()
        for i in self.I:
            for n in self.N:
                self.t[i, n] = self.rlm.add_var(f't_{i}_{n}')

        self.delta = dict()
        for l in self.L:
            self.delta[l] = self.rlm.add_var(f'delta_{l}')

        self.y = dict()
        for i in self.I:
            self.y[i] = self.rlm.add_var(f'y_{i}', type_=BINARY)

        self.z = dict()
        for j in self.J:
            self.z[j] = self.rlm.add_var(f'z_{j}', type_=BINARY)

        self.w = dict()
        for m in self.M:
            self.w[m] = self.rlm.add_var(f'w_{m}', type_=BINARY)

    def init_objective(self) -> None:
        F_y_sumover_i = 0
        G_z_sumover_j = 0
        H_w_sumover_m = 0
        C_star_x_sumover_ki = 0
        A_star_u_sumover_ij = 0
        B_star_p_sumover_jm = 0
        E_star_q_sumover_ml = 0
        V_star_t_sumover_in = 0
        P_delta_sumover_l = 0
        for i in self.I:
            F_y_sumover_i += self.F[i] * self.y[i]
        for j in self.J:
            G_z_sumover_j += self.G[j] * self.z[j]
        for m in self.M:
            H_w_sumover_m += self.H[m] * self.w[m]
        for k in self.K:
            for i in self.I:
                C_star_x_sumover_ki += self.C_star[k][i] * self.x[k, i]
        for i in self.I:
            for j in self.J:
                A_star_u_sumover_ij += self.A_star[i][j] * self.u[i, j]
        for j in self.J:
            for m in self.M:
                B_star_p_sumover_jm += self.B_star[j][m] * self.p[j, m]
        for m in self.M:
            for l in self.L:
                E_star_q_sumover_ml += self.E_star[m][l] * self.q[m, l]
        for i in self.I:
            for n in self.N:
                V_star_t_sumover_in += self.V_star[i][n] * self.t[i, n]
        for l in self.L:
            P_delta_sumover_l += self.P[l] * self.delta[l]

        objective = F_y_sumover_i + G_z_sumover_j + H_w_sumover_m + C_star_x_sumover_ki + A_star_u_sumover_ij + \
            B_star_p_sumover_jm + E_star_q_sumover_ml + \
            V_star_t_sumover_in + P_delta_sumover_l

        self.rlm.set_objective(objective, sense=MINIMIZE,
                               uncertainty_set=self.objective_unc_set)

    def init_constraints(self) -> None:
        # c1
        for l in self.L:
            q_sumover_m = 0
            for m in self.M:
                q_sumover_m += self.q[m, l]
            self.rlm.add_constraint(
                q_sumover_m + self.delta[l] >= self.D_star[l], uncertainty_set=self.demand_unc_sets[l])

        # c2
        for k in self.K:
            x_sumover_i = 0
            for i in self.I:
                x_sumover_i += self.x[k, i]
            self.rlm.add_constraint(
                x_sumover_i >= self.R_star[k], uncertainty_set=self.return_unc_sets[k])

        # c3
        for i in self.I:
            u_sumover_j = 0
            x_sumover_k = 0
            for j in self.J:
                u_sumover_j += self.u[i, j]
            for k in self.K:
                x_sumover_k += self.x[k, i]
            self.rlm.add_constraint(
                u_sumover_j - (1-self.S) * x_sumover_k <= 0)
            self.rlm.add_constraint(
                u_sumover_j - (1-self.S) * x_sumover_k >= 0)

        # c4
        for i in self.I:
            t_sumover_n = 0
            x_sumover_k = 0
            for n in self.N:
                t_sumover_n += self.t[i, n]
            for k in self.K:
                x_sumover_k += self.x[k, i]
            self.rlm.add_constraint(t_sumover_n - self.S * x_sumover_k <= 0)
            self.rlm.add_constraint(t_sumover_n - self.S * x_sumover_k >= 0)

        # c5
        for m in self.M:
            p_sumover_j = 0
            q_sumover_l = 0
            for j in self.J:
                p_sumover_j += self.p[j, m]
            for l in self.L:
                q_sumover_l += self.q[m, l]
            self.rlm.add_constraint(p_sumover_j - q_sumover_l <= 0)
            self.rlm.add_constraint(p_sumover_j - q_sumover_l >= 0)

        # c6
        for j in self.J:
            p_sumover_m = 0
            u_sumover_i = 0
            for m in self.M:
                p_sumover_m += self.p[j, m]
            for i in self.I:
                u_sumover_i += self.u[i, j]
            self.rlm.add_constraint(p_sumover_m - u_sumover_i <= 0)

        # c7
        for i in self.I:
            x_sumover_k = 0
            for k in self.K:
                x_sumover_k += self.x[k, i]
            self.rlm.add_constraint(x_sumover_k <= self.y[i] * self.C_C[i])

        # c8
        for j in self.J:
            u_sumover_i = 0
            for i in self.I:
                u_sumover_i += self.u[i, j]
            self.rlm.add_constraint(u_sumover_i <= self.z[j] * self.C_R[j])

        # c9
        for m in self.M:
            p_sumover_j = 0
            for j in self.J:
                p_sumover_j += self.p[j, m]
            self.rlm.add_constraint(p_sumover_j <= self.w[m] * self.C_E[m])

        # c10
        for n in self.N:
            t_sumover_i = 0
            for i in self.I:
                t_sumover_i += self.t[i, n]
            self.rlm.add_constraint(t_sumover_i <= self.C_D[n])

    def log_origin_model(self) -> None:
        self.rlm.log_origin_model()

    def transform(self, verbose: bool = False) -> None:
        self.rlm.transform(verbose=verbose)

    def log_robust_counterpart(self) -> None:
        self.rlm.log_robust_counterpart()

    def solve(self) -> None:
        self.rlm.solve()

    def log_solution(self) -> None:
        self.rlm.log_solution()

    def get_robust_objective_value(self) -> Union[float, int]:
        return self.rlm.get_robust_objective_value()

    def get_deterministic_model_objective_value(self) -> Union[float, int]:
        return self.rlm.get_deterministic_model_objective_value()

    def __data_objective_uncertainty_set(self, robustness: float) -> None:
        # objective uncertainty set
        obj_unc_nominal = dict()
        for k in self.K:
            for i in self.I:
                obj_unc_nominal[self.C_star[k][i]] = 47.5
        for i in self.I:
            for j in self.J:
                obj_unc_nominal[self.A_star[i][j]] = 47.5
        for j in self.J:
            for m in self.M:
                obj_unc_nominal[self.B_star[j][m]] = 47.5
        for m in self.M:
            for l in self.L:
                obj_unc_nominal[self.E_star[m][l]] = 47.5
        for i in self.I:
            for n in self.N:
                obj_unc_nominal[self.V_star[i][n]] = 47.5

        obj_shift_count = self.data_point_count
        obj_unc_shifts = list()
        for _ in range(obj_shift_count):
            shift = dict()
            for k in self.K:
                for i in self.I:
                    shift[self.C_star[k][i]] = round(
                        random.uniform(-7.5, 7.5), 4)
            for i in self.I:
                for j in self.J:
                    shift[self.A_star[i][j]] = round(
                        random.uniform(-7.5, 7.5), 4)
            for j in self.J:
                for m in self.M:
                    shift[self.B_star[j][m]] = round(
                        random.uniform(-7.5, 7.5), 4)
            for m in self.M:
                for l in self.L:
                    shift[self.E_star[m][l]] = round(
                        random.uniform(-7.5, 7.5), 4)
            for i in self.I:
                for n in self.N:
                    shift[self.V_star[i][n]] = round(
                        random.uniform(-7.5, 7.5), 4)
            obj_unc_shifts.append(shift)

        self.objective_unc_set = Box("objective_box",
                                     robustness=robustness, nominal_data=obj_unc_nominal, base_shifts=obj_unc_shifts)

    def __data_demand_uncertainty_sets(self, robustness: float) -> None:

        # demand uncertainty sets
        demand_unc_nominals = list()
        demand_unc_shiftss = list()

        for l in self.L:  # for each constraint an uncertainty set
            demand_shift_count = self.data_point_count
            demand_unc_nominals.append(dict())
            demand_unc_shiftss.append(list())

            demand_unc_nominals[l][self.D_star[l]] = 450
            for _ in range(demand_shift_count):
                shift = dict()
                shift[self.D_star[l]] = round(random.uniform(-100, 100), 4)
                demand_unc_shiftss[l].append(shift)

        self.demand_unc_sets = [Box(f"demand_box_{l}", robustness=robustness, nominal_data=demand_unc_nominals[l],
                                    base_shifts=demand_unc_shiftss[l]) for l in self.L]

    def __data_return_uncertainty_sets(self, robustness: float) -> None:
        # return uncertainty sets
        return_unc_nominals = list()
        return_unc_shiftss = list()

        for k in self.K:  # for each constraint an uncertainty set
            return_shift_count = self.data_point_count
            return_unc_nominals.append(dict())
            return_unc_shiftss.append(list())

            return_unc_nominals[k][self.R_star[k]] = 550
            for _ in range(return_shift_count):
                shift = dict()
                shift[self.R_star[k]] = round(random.uniform(-100, 100), 4)
                return_unc_shiftss[k].append(shift)

        self.return_unc_sets = [Box(f"return_box_{k}", robustness=robustness, nominal_data=return_unc_nominals[k],
                                    base_shifts=return_unc_shiftss[k]) for k in self.K]

    def __range_objective_uncertainty_set(self, robustness: float) -> None:
        obj_unc_nominal = dict()
        for k in self.K:
            for i in self.I:
                obj_unc_nominal[self.C_star[k][i]] = 47.5
        for i in self.I:
            for j in self.J:
                obj_unc_nominal[self.A_star[i][j]] = 47.5
        for j in self.J:
            for m in self.M:
                obj_unc_nominal[self.B_star[j][m]] = 47.5
        for m in self.M:
            for l in self.L:
                obj_unc_nominal[self.E_star[m][l]] = 47.5
        for i in self.I:
            for n in self.N:
                obj_unc_nominal[self.V_star[i][n]] = 47.5
        shift = dict()
        for k in self.K:
            for i in self.I:
                shift[self.C_star[k][i]] = 7.5
        for i in self.I:
            for j in self.J:
                shift[self.A_star[i][j]] = 7.5
        for j in self.J:
            for m in self.M:
                shift[self.B_star[j][m]] = 7.5
        for m in self.M:
            for l in self.L:
                shift[self.E_star[m][l]] = 7.5
        for i in self.I:
            for n in self.N:
                shift[self.V_star[i][n]] = 7.5
        obj_unc_shifts = [shift]

        self.objective_unc_set = Box("objective_box",
                                     robustness=robustness, nominal_data=obj_unc_nominal, base_shifts=obj_unc_shifts)

    def __range_demand_uncertainty_sets(self, robustness: float) -> None:
        demand_unc_nominals = list()
        demand_unc_shiftss = list()

        for l in self.L:  # for each constraint an uncertainty set
            demand_unc_nominals.append(dict())
            demand_unc_shiftss.append(list())

            demand_unc_nominals[l][self.D_star[l]] = 450
            shift = dict()
            shift[self.D_star[l]] = 100
            demand_unc_shiftss[l].append(shift)

        self.demand_unc_sets = [Box(f"demand_box_{l}", robustness=robustness, nominal_data=demand_unc_nominals[l],
                                    base_shifts=demand_unc_shiftss[l]) for l in self.L]

    def __range_return_uncertainty_sets(self, robustness: float) -> None:
        return_unc_nominals = list()
        return_unc_shiftss = list()

        for k in self.K:  # for each constraint an uncertainty set
            return_unc_nominals.append(dict())
            return_unc_shiftss.append(list())

            return_unc_nominals[k][self.R_star[k]] = 550

            shift = dict()
            shift[self.R_star[k]] = 100
            return_unc_shiftss[k].append(shift)

        self.return_unc_sets = [Box(f"return_box_{k}", robustness=robustness, nominal_data=return_unc_nominals[k],
                                    base_shifts=return_unc_shiftss[k]) for k in self.K]

    def evaluate(self, sample: int = 10, seed=5, verbose: bool = True):
        random.seed(seed)
        # make deterministic model first, and then compare performances.
        self.rlm.make_deterministic_model_using_nominal_data()
        # TODO: just pseudo code
        decisions = list(self.y.values()) + \
            list(self.z.values()) + list(self.w.values())
        comparisons = {
            "deterministic": list(),
            "robust": list()
        }
        wondered = [self.delta[l] for l in self.L]
        for _ in tqdm(range(sample)):
            deter_real, robust_real, deter_wondered, robust_wondered = self.rlm.evaluator(realization=self.__realize(),
                                                                                          decisions=decisions, wondered=wondered)
            try:
                # print(sum(deter_wondered.values()))
                # print(sum(robust_wondered.values()))
                pass
            except:
                pass
            comparisons["deterministic"].append(deter_real)
            comparisons["robust"].append(robust_real)
            if deter_real == None:
                deter_real = "infeasible"
            if robust_real == None:
                robust_real = "infeasible"
            if verbose:
                print(_)
                print("deterministic realization:", deter_real)
                print("robust realization:", robust_real)
        generate_evaluations_plot(figure_path=os.path.join(self.figure_dir, "evaluations.png"),
                                  comparisons=comparisons, title=f"{self.rlm.name} evaluations plot")

        return mean_value_of_robustization(comparisons, self.rlm.sense), improvement_of_std(comparisons, self.rlm.sense), robust_rate(comparisons)

    def log_deterministic_model(self):
        self.rlm.log_deterministic_model()

    def get_robust_solutions(self):
        return self.rlm.get_robust_solutions()

    def get_deterministic_solutions(self):
        return self.rlm.get_deterministic_solutions()

    def __realize(self):
        realization = dict()
        for D_l in self.D_star:
            realization[D_l] = round(random.uniform(450, 550), 4)
            # realization[D_l] = round(random.uniform(350, 550), 4)
            # realization[D_l] = round(random.normalvariate(450, 33), 4)
        for R_k in self.R_star:
            realization[R_k] = round(random.uniform(450, 700), 4)
            # realization[R_k] = round(random.uniform(450, 650), 4)
            # realization[R_k] = round(random.normalvariate(550, 33), 4)
        for C_k in self.C_star:
            for C_ki in C_k:
                realization[C_ki] = round(random.uniform(40, 55), 4)
                # realization[C_ki] = round(random.normalvariate(47.5, 2), 4)
        for A_i in self.A_star:
            for A_ij in A_i:
                realization[A_ij] = round(random.uniform(45, 55), 4)
                # realization[A_ij] = round(random.normalvariate(50, 1.5), 4)
        for B_j in self.B_star:
            for B_jm in B_j:
                realization[B_jm] = round(random.uniform(45, 55), 4)
                # realization[B_jm] = round(random.normalvariate(50, 1.5), 4)
        for E_m in self.E_star:
            for E_ml in E_m:
                realization[E_ml] = round(random.uniform(45, 55), 4)
                # realization[E_ml] = round(random.normalvariate(50, 1.5), 4)
        for V_i in self.V_star:
            for V_in in V_i:
                realization[V_in] = round(random.uniform(45, 55), 4)
                # realization[V_in] = round(random.normalvariate(50, 1.5), 4)
        return realization
