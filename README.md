# CPU Thermal Diffusion and Geometric Optimization

## Overview

Modern CPUs dissipate high power densities, leading to hotspots that impact performance and reliability.

This project models heat diffusion inside a CPU die using a thermal RC network derived from the heat equation. The implementation was applied to a simplified model of the **Intel Atom D2700**, using a coarse mesh for computational efficiency. The framework is modular and can be extended to finer discretizations or more complex architectures.

Developed as part of a TIPE (CPGE research project).

---

## Governing Equation

Heat diffusion is governed by:

$$
\rho c \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T) + P
$$

where:
- $\rho c$ is the volumetric heat capacity,
- $k$ the thermal conductivity,
- $P$ the power density.

---

## Thermal RC Discretization

The die is discretized into a 2D grid. Each cell is modeled as a thermal capacitance connected to its neighbors via thermal resistances.

For each node $i$:

$$
C_i \frac{dT_i}{dt}
=
\sum_{j \in V(i)} \frac{T_j - T_i}{R_{ij}}
+ P_i
- h_i A_i (T_i - T_{amb})
+ \frac{T_{sub} - T_i}{R_b}
$$

This produces a coupled system of ODEs.

---

## Crank–Nicolson Time Scheme

To ensure stability and second-order accuracy in time, the Crank–Nicolson scheme is used:

$$
\frac{C_i}{\Delta t}(T_i^{n+1} - T_i^n)
=
\frac{1}{2} \left[ F_i(T^n) + F_i(T^{n+1}) \right]
$$

After rearrangement, this leads to a linear system:

$$
A T^{n+1} = B T^n + S
$$

Properties:

- Unconditionally stable  
- Second-order accuracy in time ($O(\Delta t^2)$)  
- Energy consistent  

---

## Matrix Structure

For $N = n_x \times n_y$ nodes:

Diagonal entries:

$$
A_{ii} =
\frac{C_i}{\Delta t}
+
\frac{1}{2} \sum_{j \in V(i)} \frac{1}{R_{ij}}
+
\frac{h_i A_i}{2}
+
\frac{1}{2 R_b}
$$

Off-diagonal entries:

$$
A_{ij} = -\frac{1}{2R_{ij}}
$$

The matrix $A$ is sparse, symmetric, and positive definite.

---

## Numerical Solver

The linear system is solved using the **Conjugate Gradient (CG)** method, which is well-suited for large sparse symmetric positive definite systems.

---

## Geometric Optimization

A simulated annealing procedure is used to optimize the spatial arrangement of functional blocks in order to:

- Reduce maximum temperature
- Minimize thermal gradients

On the Intel Atom D2700 test configuration, the optimization reduced peak temperature by approximately 5°C.

---

## Reliability Impact

Using the Arrhenius law:

$$
MTTF \propto e^{\frac{E_a}{k_B T}}
$$

A temperature reduction from 56°C to 51°C yields an estimated lifetime increase of approximately 38%.

---

## Technical Stack

- Python
- NumPy
- Sparse matrix assembly
- Conjugate Gradient solver
- Scientific visualization


---

## Keywords

Heat equation, thermal RC network, Crank–Nicolson, sparse matrices, conjugate gradient, floorplanning optimization, hotspot mitigation.

