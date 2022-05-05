# Preventive Maintenance Scheduling Problem

todo: 把不同時間段的遞迴式合併轉成一般式
##  Sets
$T$: Periods taken into consideration, $T = \{0, 1, ..., |T| - 1\}$. 
> $t = 0$ is the initial state, no decisions involved.

## Parameters
### Deterministic
$C^S$: Stockout cost per unit of product.  
$C^H$: Holding cost per unit of product.  
$C^{CM}$: Cost of adopting corrective maintenance once.  
$C^{PM}$: Cost of adopting preventive maintenance once.  
$T^{PM}$: Time spent by adopting preventive maintenance once, in periods.  
$T^{CM}$: Time spent by adopting corrective maintenance once, in periods.  
$H^{\text{max}}$: Maximum holding amount.  
$H^I$: Initial holding amount.  
$B^I$: Initial backorder amount.  
$S^I$: Initial health status of the machine.  
$R$: Recovery of health status after adopting preventive maintenance.  
$Z$: Health status threshold determine CM is required or not.  
$M$: Great number.
### Uncertain
$D^*$: Deterioration rate of health status after a production period.  
$Q_t$: Demand quantity in period $t$.
### Functions
$Y(s)$: Maximum output of a machine per period at health status $s$.
> E.g. $Y(s) = 100s$.

## Decision Variables
$x_t$: Production amount in period $t$, $t \in T$  
$z^{PM}_t$: Whether PM is adopted or not in period $t$, $t \in T$  
$z^{CM}_t$: Whether CM is adopted or not in period $t$, $t \in T$  
$s_t$: Health status of the machine in period $t$, $0 \leq s \leq 1$, $t \in T$  
$z^P_t$: Whether the machine is set up or not in period $t$, $t \in T$  
$b_t$: Backorder amount in period $t$, $t \in T$  
$h_t$: Holding amount in period $t$,  $t \in T$

## Objectives
$\text{min.}$ Holding cost + Back order cost + PM cost + CM cost

Holding cost $\displaystyle{=C^H \cdot \sum_{t \in T} h_t}$  
Back order cost $\displaystyle{=C^S \cdot \sum_{t \in T}b_t}$  
PM cost $\displaystyle{=C^{PM} \cdot \sum_{t \in T}z^{PM}_t}$  
CM cost $\displaystyle{=C^{CM} \cdot \sum_{t \in T}z^{CM}_t}$

## Constraints
c1: States initialization.  

$x_0 = z^{PM}_0 = z^{CM}_0 = z^P_0 = 0$  
$h_0 = H^I$  
$b_0 = B^I$  
$s_0 = S^I$

---
c2: Exclusivity of PM, CM, and production.  
<!-- Once if starting PM or CM, the following days cannot start PM or CM anymore. -->
$\displaystyle{(1-z^{PM}_t) \cdot M \geq \sum_{t^\prime=t+1,\\t^\prime \in T}^{t+T^{PM}-1}z^{PM}_{t^\prime} + z^{CM}_{t^\prime}, \forall t \in T}$  
$\displaystyle{(1-z^{CM}_t) \cdot M \geq \sum_{t^\prime=t+1,\\t^\prime \in T}^{t+T^{CM}-1}z^{PM}_{t^\prime} + z^{CM}_{t^\prime}, \forall t \in T}$  
$\displaystyle{(1-z^{PM}_t) \cdot M \geq \sum_{t^\prime=t,\\t^\prime \in T}^{t+T^{PM}-1}x_{t^\prime}, \forall t \in T}$  
$\displaystyle{(1-z^{CM}_t) \cdot M \geq \sum_{t^\prime=t,\\t^\prime \in T}^{t+T^{CM}-1}x_{t^\prime}, \forall t \in T}$

---
c3: Machine can only be resting, starting PM, starting CM, producing, or  maintaining.

$z^{PM}_t + z^{CM}_t + z^P_t \leq 1, \forall t \in T$

---
c4: Adopt CM if the health status of the machine less than the threshold.  

$z^{CM}_t \geq (Z - s_t),~\forall t \in T$  
> CM is required if $Z - s_t > 0$.

---
c5: Health status of the machine recover after starting PM or CM.

$z^{CM}_t \leq s_{t+1}, \forall t \in T, t \neq |T| - 1$  
$s_{t+1} \leq (s_t + R) + M(1 - z^{PM}_t), \forall t \in T, t \neq |T| - 1$

---
c6: Health status of the machine deteriorate in a rate if is setup.

<!-- $s_{t+1} \geq s_{t} - (1 - z^P_t) \cdot M, \forall t \in T, t \neq |T| - 1$ -->

$s_{t+1} \leq D^* s_{t} + (1 - z^P_t) \cdot M, \forall t \in T, t \neq |T| - 1$ 

---
c7: Determine setup or not. 

$M z^P_t \geq x_t, \forall t \in T$

$x_t \geq z^P_t, \forall t \in T$

---
c8: If not producing or adopting PM / CM, the machine stage should stay unchanged.

$s_{t+1} \geq s_{t} - (z^P_t + z^{PM}_t + z^{CM}_t) \cdot M, \forall t \in T, t \neq |T|-1$  
$s_{t+1} \leq s_{t} + (z^P_t + z^{PM}_t + z^{CM}_t) \cdot M, \forall t \in T, t \neq |T|-1$

---
c9: Determine the largest yield amount.

$x_t \leq Y(s_t), \forall t \in T$

---
c10: Maintainence of holding, stock out, and supply / demand balance.

$\displaystyle{h_{t} \leq x_t + h_{t-1} - (Q_t + b_{t-1}) + b_t , \forall t \in T, t \neq 0}$

---
c11: Maximum inventory level.

$h_t \leq H^{\text{max}}, \forall t \in T$

---
c12: Domain of the variables.

$x_t \in \mathbb{N}, \forall t \in T$  
$z^{PM}_t \in \{0,1\}, \forall t \in T$  
$z^{CM}_t \in \{0,1\}, \forall t \in T$  
$s_t \in [0, 1], t \in T$  
$z^P_t \in \{0, 1\}, t \in T$  
$b_t \in \mathbb{N}, t \in T$  
$h_t \in \mathbb{N}, t \in T$

--- 
candidate
## Sets
<!-- $P$: Set of time periods taken into consideration   -->
$L$: Set of production lines  
$O$: Set of orders

## Parameters
$T^E_o$: Earliest start time of producing order $o$  
$T^L_o$: Lastest finish time of producing order $o$  
$Q_o$: Required quantity of order $o$  
$Y_l$: Yielding rate of production line $l$  
$M$: Great number  
$P$: Penalty of unit time lateness  


## Variables
$x_{lo}$: Start time of order $o$ in production line $l$  
$z_{lo}$: Whether the order $o$ is assigned to the production line $l$  

## Objective

## Constraints
Every order should be assigned to a production line  
$x_{lo} \leq Mz_{lo},~ \forall l \in L,~o \in O$  
$\displaystyle{\sum_{l \in L} z_{lo} = 1,~\forall o \in O}$  


Production exclusive  
$x_{l^\prime o} \geq  or,~\forall l^\prime \in L,~l \in L \setminus \{l^\prime\},~o \in O$

