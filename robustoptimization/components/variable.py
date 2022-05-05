from sympy import Symbol
from numpy import Inf


class Variable(Symbol):

    def __init__(self, name: str) -> None:
        self.type_ = None
        self.lb = None
        self.ub = None
