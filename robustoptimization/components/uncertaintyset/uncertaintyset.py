from typing import Dict, List
from robustoptimization.components.uncertainparameter import UncertainParameter


class UncertaintySet:
    def __init__(self, name: str,
                 nominal_data: Dict[UncertainParameter, float],
                 base_shifts: List[Dict[UncertainParameter, float]],
                 robustness: float = 1.0) -> None:
        self.name = name
        self.robustness = robustness
        self.nominal_data = nominal_data
        self.base_shifts = base_shifts

        # self.robustness = robustness
        # # self.__variable_count = None
        # self.__base_shift_count = None
        # self.nominal_data = nominal_data
        # # key: Parameter object, value: determined Parameter object
        # self.base_shifts = base_shifts
        # # [dict(key: Parameter object, value: determined Parameter object)]

    # def set_variable_count(self, variable_count):
    #     self.__variable_count = variable_count
    #     if len(self.nominal_data) != self.__variable_count:
    #         raise ValueError("variable count doesn't match nominal data.")
    #     for data in self.base_shifts:
    #         if len(data) != self.__variable_count:
    #             raise ValueError(
    #                 "variable count doesn't match base shift data.")

    # def add_base_shift(self, base_shift):
    #     self.__base_shift_count = len(self.base_shifts)
    #     self.base_shifts.append(base_shift)

    # def get_base_shift_count(self):
    #     return self.__base_shift_count

    # def get_human_readable(self):
    #     readable = ""
    #     readable += "\t\t\trobustness = {}\n".format(self.robustness)
    #     readable += "\t\t\tnominal data:\n"
    #     for v in self.nominal_data:
    #         coe = self.nominal_data[v]
    #         readable += "\t\t\t\t{}: {}".format(coe.name, coe.value)
    #     readable += "\n"
    #     readable += "\t\t\tbase shifts:\n"
    #     for i, data in enumerate(self.base_shifts):
    #         readable += "\t\t\t\tdata_{}:\n".format(i)
    #         for v in data:
    #             readable += "\t\t\t\t\t{}: {}\n".format(
    #                 data[v].name, data[v].value)

    #     return readable

    # def display(self):
    #     print(self.get_human_readable())
