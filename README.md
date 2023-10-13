# Robust optimization package
## Setup
  1. new virtual environment `python3 -m venv env`
  2. activate virtual environment: `source env/bin/activate` (windows: `env\Script\activate`)
  3. install requirements: `pip install requirements.txt`

- To solve non-linear convex model please install `ipopt` in your computer first (for non-linear uncertainty set)

## Explanations
- Robust optimization is a methodology handles uncertainty without distributional assumptions
- You can choose your own uncertainty set (must be convex), supported sets are listed below:
  - Box: `robustoptimization.components.uncertaintyset.box`
  - Ball: `robustoptimization.components.uncertaintyset.ball`
  - others: refer to the reference `./refs` and inherit `robustoptimization.components.uncertaintyset.UncertaintySet` on your own
- Metrics for RO solution qualities are defined in `./robustoptimization/utils/metrics.py`, including:
  - `mean_value_of_robustization`: The mean objective value improvement of RO compared to deterministic optimization 
  - `improvement_of_std`: The objective std improvement of RO compared to deterministic optimization 
  - `robust_rate`: Proportion of solutions feasible using RO but infeasible using deterministic optimization

## Examples

- supply chain network model
  - mathematical formulation provided at `./scn.md`
  - code definition (OOP wrapper) provided at `supplychainnetworkmodel.py`
  - entry point: `python3 main.py --scn`
- machine scheduling model
  - mathematical formulation provided at `./scheduling.md`
  - code definition (OOP wrapper) provided at `./schedulingmodel.py`
  - entry point: `python3 main.py --sch`


## Future works
- only variable >= 0 supported
- check dim consistency  
- check variable and parameter naming  
- math simplification  
- box robustness parameter  
- simplify parameter object to uncertain param  
- sympy  https://stackoverflow.com/questions/30225348/how-to-rearrange-sympy-expressions-containing-a-relational-operator
- bind uncertainty set with constraint as class
- small cases and big ones to demonstrate package util.