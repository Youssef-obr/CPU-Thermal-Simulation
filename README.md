# CPU Thermal Diffusion and Geometric Optimization

## Overview

Modern CPUs dissipate high thermal power densities, leading to the formation of hotspots that degrade performance and reliability.

This project models and simulates heat diffusion inside a CPU die using a thermal RC network derived from the heat equation. The framework was applied to a test architecture inspired by the **Intel Atom D2700**, using a simplified spatial mesh for computational efficiency. The implementation is modular and can be extended to finer meshes or more complex architectures.

Developed as part of a TIPE (CPGE research project).

---

## Physical Model

### Heat Equation

Heat diffusion in the die is governed by:

\[
\rho c \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T) + P
\]

where:
- \( \rho c \) : volumetric heat capacity  
- \( k \) : thermal conductivity  
- \( P \) : power density  

---

### Thermal RC Discretization

The die is discretized into a 2D grid of thermal cells.

For each node \( i \), the energy balance writes:

\[
C_i \frac{dT_i}{dt}
=
\sum_{j \in V(i)} \frac{T_j - T_i}{R_{ij}}
+ P_i
- h_i A_i (T_i - T_{amb})
+ \frac{T_{sub} - T_i}{R_b}
\]

with:

- \( C_i \) : thermal capacitance  
- \( R_{ij} \) : lateral thermal resistance  
- \( R_b \) : resistance to substrate  
- \( h_i \) : convection coefficient  
- \( P_i \) : dissipated power  

This forms a system of coupled ODEs.

---

## Crank–Nicolson Time Integration

To ensure stability and second-order accuracy in time, the Crank–Nicolson scheme is used.

\[
\frac{C_i}{\Delta t}(T_i^{n+1} - T_i^n)
=
\frac{1}{2}
\left(
F_i(T^n) + F_i(T^{n+1})
\right)
\]

where \( F_i(T) \) represents the spatial diffusion and source terms.

After rearrangement, this leads to a linear system:

\[
A T^{n+1} = B T^n + S
\]

Properties:

- Second-order accuracy in time: \( O(\Delta t^2) \)
- Unconditionally stable
- Energy-consistent

---

## Matrix Structure

For a grid of size \( N = n_x \times n_y \):

- \( A \) is sparse, symmetric, positive definite  
- \( B \) is sparse  
- Only neighboring nodes are coupled  

Diagonal entries:

\[
A_{ii} =
\frac{C_i}{\Delta t}
+
\frac{1}{2} \sum_{j \in V(i)} \frac{1}{R_{ij}}
+
\frac{h_i A_i}{2}
+
\frac{1}{2 R_b}
\]

Off-diagonal entries:

\[
A_{ij} = -\frac{1}{2R_{ij}}
\]

---

## Numerical Solver

The linear system is solved using the **Conjugate Gradient (CG)** method:

- Suitable for symmetric positive definite matrices
- Efficient for large sparse systems
- Memory efficient

---

## Geometric Optimization

Thermal performance is improved by optimizing the spatial arrangement of functional blocks using simulated annealing.

Objective:
- Reduce maximum temperature
- Minimize thermal gradients

A temperature reduction of approximately 5°C was achieved on the Intel Atom D2700 test configuration.

---

## Reliability Impact

Using the Arrhenius law:

\[
MTTF \propto e^{\frac{E_a}{k_B T}}
\]

A temperature reduction from 56°C to 51°C leads to:

- ~38% increase in expected lifetime.

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

