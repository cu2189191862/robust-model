import sympy
from sympy.core.relational import Relational
from typing import Type, List

from robustoptimization.components.variable import Variable

mult_by_minus_one_map = {
    None: '==',
    '==': '==',
    'eq': '==',
    '!=': '!=',
    '<>': '!=',
    'ne': '!=',
    '>=': '<=',
    'ge': '<=',
    '<=': '>=',
    'le': '>=',
    '>': '<',
    'gt': '<',
    '<': '>',
    'lt': '>',
}


def inequality_rearranger(ineq: Type[Relational], vars: List[Variable]) -> Relational:
    l = ineq.lhs
    r = ineq.rhs
    op = ineq.rel_op
    all_on_left = l - r

    coeff_dict = all_on_left.as_coefficients_dict()
    var_types = coeff_dict.keys()
    new_rhs = sympy.sympify(0)
    for s in var_types:
        not_variable_term = len(s.free_symbols.intersection(vars)) == 0
        if not_variable_term:
            all_on_left = all_on_left - coeff_dict[s] * s
            new_rhs = new_rhs - coeff_dict[s] * s
    if op != "<=" and op != "lt":
        all_on_left = all_on_left * -1
        new_rhs = new_rhs * -1
        op = mult_by_minus_one_map[op]
    return Relational(all_on_left, new_rhs, op)
