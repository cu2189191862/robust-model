from typing import Union, Type, Tuple, List
from datetime import datetime
from numpy import tri
from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition
from sympy import Mul
from sympy.core.add import Add
from sympy.core.relational import Relational
from robustoptimization.components.variable import Variable
from robustoptimization.components.uncertaintyset.uncertaintyset import UncertaintySet
from robustoptimization.components.uncertaintyset.box import Box
from robustoptimization.components.uncertaintyset.ball import Ball
from robustoptimization.utils.rearranger import inequality_rearranger
from robustoptimization.utils.constants import *


import sys


class RobustLinearModel:
    '''Robust linear optimization model.
    min. z
    Ax <= b: A, b in U_i for each row i.

    Attributes:
        name (str): name of the model.
        __established_time (str): established time of the model.
        __sense (str): "minimize" or "maximize" the objective.
        __objective (int): objective of the model, the variable must exist in __x and bounded by the constraint part.
        __A ([[Parameter]]): parameters for the constraint parts left hand side.
        __b ([Parameter]): parameters for the constraint parts right hand side.
        __x ([Variable]): variables in the model.
        __U ([UncertaintySet]): uncertainty sets for each row.
        __log_writer (_io.TextIOWrapper): logging file handler.
    '''

    def __init__(self, log_path: str, name: str = None, solver: str = GUROBI) -> None:
        '''Constructor of RobustLinearModel.
        Args:
            log_path (str): path of the logging file lies.
        '''
        self.name = name
        self.__established_time = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
        self.solver = solver
        # human readable
        self.vars = list()
        self.constraints = list()
        self.sense = MINIMIZE
        self.objective = None
        # abstract part
        self.__sense = self.sense
        self.__z = None
        self.__A = None
        self.__b = list()
        self.__x = None
        self.__U = list()
        self.__log_writer = open(log_path, "w")
        self.robust_counterpart = None

    def __del__(self) -> None:
        '''Destructor of RobustLinearModel.
        '''
        self.__log_writer.close()

    def add_var(self, var_name: str, lb: float = 0.0, ub: float = None, type_: str = CONTINUOUS) -> Variable:
        new_var = Variable(var_name)
        new_var.type_ = type_
        new_var.lb = lb
        new_var.ub = ub
        self.vars.append(new_var)
        return new_var

    def add_constraint(self, constraint: Type[Relational], uncertainty_set: Union[Type[UncertaintySet], None] = None) -> None:
        # TODO: link uncertainty set with constraint using Constraint object
        if uncertainty_set != None and not issubclass(type(uncertainty_set), UncertaintySet):
            raise TypeError("Only subclass of UncertaintySet can be passed.")
        inequality_rearranged = inequality_rearranger(
            constraint, vars=self.vars)
        self.constraints.append(inequality_rearranged)
        self.__U.append(uncertainty_set)

    def set_objective(self, objective: Type[Add], sense: str = MINIMIZE, uncertainty_set: Union[Type[UncertaintySet], None] = None):
        if uncertainty_set != None and not issubclass(type(uncertainty_set), UncertaintySet):
            raise TypeError("Only subclass of UncertaintySet can be passed.")
        self.objective = (objective, uncertainty_set)
        self.sense = sense

    def transform(self, verbose: bool = False) -> None:
        self.__abstract_model()
        if verbose:
            self.__log_abstracted_model()
        self.__make_robust_counterpart()

    def __abstract_model(self) -> None:
        self.__abstract_objective()
        self.__abstract_x()
        self.__abstract_b()
        self.__abstract_A()

    def __abstract_objective(self) -> None:
        # TODO: separate non user defined constraint????
        obj_z = self.add_var("obj_z", type_=CONTINUOUS)
        self.__z = self.vars[-1]
        objective = self.objective[0]
        unc_set = self.objective[1]
        self.__sense = self.sense
        if self.sense == MAXIMIZE:
            raise NotImplementedError(
                "Here is a critical bug!!!!!!\nTo be solved in next release.")
            self.add_constraint(obj_z <= objective, uncertainty_set=unc_set)
        elif self.sense == MINIMIZE:
            self.add_constraint(obj_z >= objective, uncertainty_set=unc_set)
        else:
            raise ValueError(f"Non-supported sense {self.sense}.")

    def __abstract_x(self) -> None:
        self.__x = self.vars

    def __abstract_b(self) -> None:
        for constraint in self.constraints:
            # TODO: handle in formal way
            try:
                self.__b.append(float(constraint.rhs))
            except:
                self.__b.append(constraint.rhs)

    def __abstract_A(self) -> None:
        self.__A = list()
        vars = self.vars
        for constraint in self.constraints:
            row_coefs = list()
            raw_coe_dict = constraint.lhs.as_coefficients_dict()
            coe_dict = dict()
            for term in raw_coe_dict.keys():
                free_symbols = term.free_symbols
                if len(free_symbols) != 1:
                    v = free_symbols.intersection(vars).pop()
                    # TODO: raise error if exist unc*unc case
                    unc_params = free_symbols.difference(vars)
                    coe_dict[v] = raw_coe_dict[term]  # scalar param
                    for unc_param in unc_params:
                        coe_dict[v] *= unc_param  # uncertain param
                else:
                    coe_dict[free_symbols.pop()] = raw_coe_dict[term]

            for v in vars:
                if v in coe_dict:
                    try:
                        row_coefs.append(float(coe_dict[v]))
                    except:
                        row_coefs.append(coe_dict[v])
                else:
                    row_coefs.append(0)
            self.__A.append(row_coefs)

    def solve(self) -> None:
        opt = SolverFactory(self.solver)  # ipopt, gurobi, baron
        # print(value(self.robust_counterpart.x_0))
        opt.solve(self.robust_counterpart)

    def __make_robust_counterpart(self) -> None:
        # pyomo model initialize
        self.robust_counterpart = ConcreteModel(
            f"{self.name}: robust counterpart")
        # add origin decision variables
        for n, var in enumerate(self.__x):
            if var.type_ == CONTINUOUS:
                domain = NonNegativeReals
            elif var.type_ == INTEGER:
                domain = PositiveIntegers
            elif var.type_ == BINARY:
                domain = Binary
            else:
                raise ValueError(f"not supported variable type {var.type_}")
            exec(f"self.robust_counterpart.{var} = \
                Var(bounds=({var.lb}, {var.ub}), within={domain})")
        # objective
        exec(f"self.robust_counterpart.objective = \
            Objective(expr=self.robust_counterpart.obj_z, sense={self.__sense})")

        constraint_count = len(self.__U)
        for m in range(constraint_count):
            type_ = type(self.__U[m])
            if type_ == Box:
                self.__box_transform(m)
            elif type_ == Ball:
                self.__ball_transform(m)
            elif self.__U[m] == None:
                self.__none_transform(m)

    def __box_transform(self, constraint_idx: int) -> None:
        trans_count = 0
        unc_set = self.__U[constraint_idx]
        b = self.__b[constraint_idx]
        for l, base_shift in enumerate(unc_set.base_shifts):
            # induced variable u_l
            exec(f"self.robust_counterpart.c{constraint_idx}_u_{l} = \
                Var(within=NonNegativeReals)")
            a_shift_x = ""
            for n, x in enumerate(self.__x):
                param = self.__A[constraint_idx][n]  # parameter A[m][n]
                if self.__is_uncertain(param):  # uncertain parameter
                    coef_dict = param.as_coefficients_dict()
                    unc_param = list(coef_dict.keys())[0]
                    shift_value = base_shift[unc_param] * coef_dict[unc_param]
                else:  # scalar parameter
                    shift_value = 0
                a_shift_x += f" + {shift_value} * self.robust_counterpart.{x}"
            if self.__is_uncertain(b):  # uncertain parameter
                coef_dict = b.as_coefficients_dict()
                unc_param = list(coef_dict.keys())[0]
                b_shift = base_shift[unc_param] * coef_dict[unc_param]
            else:  # scalar parameter
                b_shift = 0
            exec(f"self.robust_counterpart.c{constraint_idx}_trans_{trans_count} = \
                Constraint(expr=\
                    -self.robust_counterpart.c{constraint_idx}_u_{l} <= \
                    {a_shift_x} - {b_shift})")
            trans_count += 1
            exec(f"self.robust_counterpart.c{constraint_idx}_trans_{trans_count} = \
                Constraint(expr=\
                    self.robust_counterpart.c{constraint_idx}_u_{l} >= \
                    {a_shift_x} - {b_shift})")
            trans_count += 1
        a_nominal_x = ""
        for n, x in enumerate(self.__x):
            param = self.__A[constraint_idx][n]
            if self.__is_uncertain(param):  # uncertain parameter
                coef_dict = param.as_coefficients_dict()
                unc_param = list(coef_dict.keys())[0]
                nominal_value = unc_set.nominal_data[unc_param] * \
                    coef_dict[unc_param]
            else:  # scalar parameter
                nominal_value = param
            a_nominal_x += f" + {nominal_value} * self.robust_counterpart.{x}"
        sum_u = ""
        for l in range(len(unc_set.base_shifts)):
            sum_u += f" + self.robust_counterpart.c{constraint_idx}_u_{l}"
        if self.__is_uncertain(b):  # uncertain parameter
            coef_dict = b.as_coefficients_dict()
            unc_param = list(coef_dict.keys())[0]
            b_nominal = unc_set.nominal_data[unc_param] * coef_dict[unc_param]
        else:  # scalar parameter
            b_nominal = b
        exec(f"self.robust_counterpart.c{constraint_idx}_trans_{trans_count} = \
            Constraint(expr=\
                {a_nominal_x} + {unc_set.robustness} * {sum_u} <= {b_nominal})")
        trans_count += 1

    def __ball_transform(self, constraint_idx: int) -> None:
        a_nominal_x = ""
        unc_set = self.__U[constraint_idx]
        b = self.__b[constraint_idx]
        for n, x in enumerate(self.__x):
            param = self.__A[constraint_idx][n]
            if self.__is_uncertain(param):
                coef_dict = param.as_coefficients_dict()
                unc_param = list(coef_dict.keys())[0]
                nominal_value = unc_set.nominal_data[unc_param] * \
                    coef_dict[unc_param]
            else:
                nominal_value = param
            a_nominal_x += f" + {nominal_value} * self.robust_counterpart.{x}"

        squared_sum = ""
        for l, base_shift in enumerate(unc_set.base_shifts):
            a_shift_x = ""
            for n, x in enumerate(self.__x):
                param = self.__A[constraint_idx][n]
                if self.__is_uncertain(param):
                    coef_dict = param.as_coefficients_dict()
                    unc_param = list(coef_dict.keys())[0]
                    shift_value = base_shift[unc_param] * coef_dict[unc_param]
                else:
                    shift_value = 0
                a_shift_x += f" + {shift_value} * self.robust_counterpart.{x}"
            if self.__is_uncertain(b):
                coef_dict = b.as_coefficients_dict()
                unc_param = list(coef_dict.keys())[0]
                b_shift = base_shift[unc_param] * coef_dict[unc_param]
            else:  # scalar parameter
                b_shift = 0
            a_shift_x_minus_b = a_shift_x + f" - {b_shift}"
            squared_sum += f" + ({a_shift_x_minus_b})**2"
        if self.__is_uncertain(b):  # uncertain parameter
            coef_dict = b.as_coefficients_dict()
            unc_param = list(coef_dict.keys())[0]
            b_nominal = unc_set.nominal_data[unc_param] * coef_dict[unc_param]
        else:  # scalar parameter
            b_nominal = b
        exec(f"self.robust_counterpart.c{constraint_idx}_trans = \
            Constraint(expr=\
                {a_nominal_x} + {unc_set.robustness} * ({squared_sum})**0.5 <= {b_nominal})")

    def __none_transform(self, constraint_idx: int) -> None:
        ax = ""
        for n, x in enumerate(self.__x):
            ax += f" + {self.__A[constraint_idx][n]} * self.robust_counterpart.{x}"
        exec(f"self.robust_counterpart.c{constraint_idx} = \
            Constraint(expr=\
                {ax} <= {self.__b[constraint_idx]})")

    def __log(self, log_str: str) -> None:
        '''Logging interface the model.
        Args:
            log_str (str): log string to be written in log file.
        '''
        self.__log_writer.write(log_str)

    def get_robust_solutions(self):
        solutions = dict()
        for var in self.__x:
            exec(f"solutions[var] = value(self.robust_counterpart.{var})")
        return solutions

    def get_deterministic_solutions(self):
        solutions = dict()
        for var in self.__x:
            exec(f"solutions[var] = value(self.deterministic_model.{var})")
        return solutions

    def log_origin_model(self) -> None:
        self.__log(
            "\n****************************Origin model*********************************\n")
        # meta info.
        self.__log("Model name: {}\n".format(self.name))
        self.__log("Established time: {}\n".format(self.__established_time))

        # objective
        self.__log("Objective:\n")
        self.__log(f"\t{self.sense}:\n")
        if self.objective[1] != None:
            self.__log(f"\t{self.objective[0]}, {self.objective[1].name}\n")
        else:
            self.__log(f"\t{self.objective[0]}\n")
        # constraints
        self.__log("Constraints:\n")
        for m, constraint in enumerate(self.constraints):
            if self.__U[m] != None:
                self.__log(f"\t{constraint}, {self.__U[m].name}\n")
            else:
                self.__log(f"\t{constraint}\n")

        self.__log(
            "\n****************************End of origin model*********************************\n")

    def __log_abstracted_model(self) -> None:
        self.__log(
            "\n****************************Abstracted model*********************************\n")
        self.__log("Objective:\n")
        self.__log(f"\t{self.__sense}: {self.__z}\n")

        self.__log("Constraints:\n")
        for m in range(len(self.__A)):
            self.__log("\t")
            for n in range(len(self.__A[m])):
                ele = self.__A[m][n]
                self.__log(f"{ele}{self.__x[n]}\t")
                # self.__log(f"{str(ele)[:2]}{self.__x[n]}({type(ele)})\t")
            # self.__log(f" <= {self.__b[m]}({type(self.__b[m])})")
            self.__log(f" <= {self.__b[m]}")
            if self.__U[m] != None:
                self.__log(f"\t{self.__U[m].name}")
            self.__log("\n")
        self.__log(
            "\n****************************End of abstracted model*********************************\n")

    def log_robust_counterpart(self) -> None:
        self.__log(
            "\n****************************Robust counterpart*********************************\n")
        # set log stream to file
        tmp_stream = sys.stdout
        sys.stdout = self.__log_writer
        # log robust counterpart
        self.robust_counterpart.pprint()
        sys.stdout = tmp_stream
        self.__log(
            "\n****************************End of robust counterpart*********************************\n")

    def log_solution(self) -> None:
        self.__log(
            "\n****************************Solution*********************************\n")
        # set log stream to file
        tmp_stream = sys.stdout
        sys.stdout = self.__log_writer
        # log robust counterpart
        self.robust_counterpart.display()
        # # write lp file
        # self.robust_counterpart.write("test.mps")
        # self.robust_counterpart.write("test.lp")
        sys.stdout = tmp_stream
        self.__log(
            "\n****************************End of solution*********************************\n")

    def get_robust_objective_value(self) -> Union[int, float]:
        return value(self.robust_counterpart.objective)

    def get_deterministic_model_objective_value(self) -> Union[int, float]:
        return value(self.deterministic_model.objective)

    def make_deterministic_model_using_nominal_data(self):
        # generate deterministic model based on nominal data
        self.__generate_deterministic_model_using_nominal_data()
        # self.log_deterministic_model()
        opt = SolverFactory(self.solver)
        opt.solve(self.deterministic_model)
        self.log_deterministic_model_solution()

    def evaluator(self, realization, decisions, wondered=list()):
        '''
        parameters:
        realization: uncertainty revealed 的該 scenario 下的所有參數值
        decisions: 有哪些 variables 是固定決策
        wondered([var]): 想要 evaluator 回傳的 varaibles, obj_z (目標式值) 是預設提供的
        returns:
        deter_obj(float)
        robust_obj(float)
        robust_wondered(dict[var, float])
        deterministic_wondered(dict[var, float])
        '''
        # 同樣的一個rlm，可以做好幾次realization，表示在各種情況下這組解的品質
        # 丟一組realized的參數進來，回傳
        # deterministic nominal model和robust model各自的objective value

        # use the decisions from robust model
        deterministic_realized_obj, deterministic_realized_wondered = self.__realize_model(
            realization, decisions, self.deterministic_model, wondered)

        # use the decisions from deterministic model
        robust_realized_obj, robust_realized_wondered = self.__realize_model(
            realization, decisions, self.robust_counterpart, wondered)

        return deterministic_realized_obj, robust_realized_obj, deterministic_realized_wondered, robust_realized_wondered

    def __realize_model(self, realization, decisions, model, wondered):
        realization_model = ConcreteModel(f"{model.name} realization")
        wondered_dict = dict()
        # variables
        for var in self.__x:
            if var.type_ == CONTINUOUS:
                domain = NonNegativeReals
            elif var.type_ == INTEGER:
                domain = PositiveIntegers
            elif var.type_ == BINARY:
                domain = Binary
            else:
                raise ValueError(f"not supported variable type {var.type_}")
            exec(f"realization_model.{var} = \
                Var(bounds=({var.lb}, {var.ub}), within={domain})")
            if var in wondered:
                exec(f"wondered_dict[var] = realization_model.{var}")

        # objective
        exec(f"realization_model.objective = \
            Objective(expr=realization_model.obj_z, sense={self.__sense})")

        # constraints
        constraint_count = len(self.__U)
        for m in range(constraint_count):
            b = self.__b[m]
            ax = ""
            for n, x in enumerate(self.__x):
                param = self.__A[m][n]
                if self.__is_uncertain(param):  # uncertain parameter
                    coef_dict = param.as_coefficients_dict()
                    unc_param = list(coef_dict.keys())[0]
                    param = coef_dict[unc_param] * realization[unc_param]
                if x in decisions:
                    ax += f" + {param} * value(model.{x})"
                else:
                    ax += f" + {param} * realization_model.{x}"

            if self.__is_uncertain(b):
                coef_dict = b.as_coefficients_dict()
                unc_param = list(coef_dict.keys())[0]
                b = coef_dict[unc_param] * realization[unc_param]
            trivial = eval(f"(type({ax} <= {b}) == bool)")
            if not trivial:
                exec(
                    f"realization_model.c{m} = Constraint(expr={ax} <= {b})")

        self.__log(
            f"\n****************************Realized {realization_model.name}*********************************\n")
        # set log stream to file
        tmp_stream = sys.stdout
        sys.stdout = self.__log_writer
        # log deterministic model
        realization_model.pprint()
        sys.stdout = tmp_stream
        self.__log(
            f"\n****************************End realized {realization_model.name}*********************************\n")
        opt = SolverFactory(self.solver)
        results = opt.solve(realization_model)

        if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
            for var in wondered_dict:
                if var not in decisions:
                    wondered_dict[var] = value(wondered_dict[var])
                else:
                    exec(f"wondered_dict[var] = value(model.{var})")
            obj_value = value(realization_model.obj_z)
            return obj_value, wondered_dict
        return None, None

    def log_deterministic_model_solution(self) -> None:
        self.__log(
            "\n****************************Deterministic model solution*********************************\n")
        # set log stream to file
        tmp_stream = sys.stdout
        sys.stdout = self.__log_writer
        # log robust counterpart
        self.deterministic_model.display()
        # # write lp file
        # self.deterministic_model.write("test.mps")
        # self.deterministic_model.write("test.lp")
        sys.stdout = tmp_stream
        self.__log(
            "\n****************************End of deterministic model solution*********************************\n")

    def __generate_deterministic_model_using_nominal_data(self):
        # pyomo model initialize
        self.deterministic_model = ConcreteModel(
            f"{self.name}: deterministic_model")
        # add origin decision variables
        for n, var in enumerate(self.__x):
            if var.type_ == CONTINUOUS:
                domain = NonNegativeReals
            elif var.type_ == INTEGER:
                domain = PositiveIntegers
            elif var.type_ == BINARY:
                domain = Binary
            else:
                raise ValueError(f"not supported variable type {var.type_}")
            exec(f"self.deterministic_model.{var} = \
                Var(bounds=({var.lb}, {var.ub}), within={domain})")
        # objective
        exec(f"self.deterministic_model.objective = \
            Objective(expr=self.deterministic_model.obj_z, sense={self.__sense})")

        constraint_count = len(self.__U)
        for m in range(constraint_count):
            if self.__U[m] == None:
                self.__nominal_none_transform(m)
            else:
                self.__nominal_transform(m)

    def __nominal_none_transform(self, constraint_idx: int) -> None:
        ax = ""
        for n, x in enumerate(self.__x):
            ax += f" + {self.__A[constraint_idx][n]} * self.deterministic_model.{x}"
        exec(f"self.deterministic_model.c{constraint_idx} = \
            Constraint(expr=\
                {ax} <= {self.__b[constraint_idx]})")

    def __nominal_transform(self, constraint_idx: int):
        unc_set = self.__U[constraint_idx]
        b = self.__b[constraint_idx]
        a_nominal_x = ""
        for n, x in enumerate(self.__x):
            param = self.__A[constraint_idx][n]
            if type(param) != int and type(param) != float:  # uncertain parameter
                coef_dict = param.as_coefficients_dict()
                unc_param = list(coef_dict.keys())[0]
                nominal_value = unc_set.nominal_data[unc_param] * \
                    coef_dict[unc_param]
            else:  # scalar parameter
                nominal_value = param

            a_nominal_x += f" + {nominal_value} * self.deterministic_model.{x}"
        if type(b) != float and type(b) != int:  # uncertain parameter
            coef_dict = b.as_coefficients_dict()
            unc_param = list(coef_dict.keys())[0]
            b_nominal = unc_set.nominal_data[unc_param] * coef_dict[unc_param]
        else:  # scalar parameter
            b_nominal = b

        exec(
            f"self.deterministic_model.c{constraint_idx} = Constraint(expr={a_nominal_x} <= {b_nominal})")

    def log_deterministic_model(self):
        self.__log(
            "\n****************************Deterministic model*********************************\n")
        # set log stream to file
        tmp_stream = sys.stdout
        sys.stdout = self.__log_writer
        # log deterministic model
        self.deterministic_model.pprint()
        sys.stdout = tmp_stream
        self.__log(
            "\n****************************End deterministic model*********************************\n")

    def __is_uncertain(self, param):
        return type(param) != int and type(param) != float
