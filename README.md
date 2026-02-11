# CPU Thermal Diffusion and Geometric Optimization

## Overview

Modern CPUs dissipate high thermal power densities, leading to the formation of hotspots that negatively impact performance and reliability. Efficient thermal management is therefore critical to maintain computational efficiency while minimizing cooling costs.

This project investigates the modeling and optimization of heat diffusion within a CPU die. The objective is to simulate temperature distribution, analyze hotspot formation, and optimize geometric configurations to improve thermal dissipation.

This work was developed as part of a TIPE (CPGE research project).

---

## Problem Statement

How can thermal diffusion within a CPU be modeled and optimized in order to:

- Limit the formation of hotspots,
- Prevent performance degradation,
- Improve overall thermal efficiency?

What numerical models and optimization methods allow for a more homogeneous temperature distribution under realistic CPU constraints?

---

## Methodology

### 1. Thermal Modeling

The CPU die is discretized into a 2D thermal grid. Each cell is modeled as a:

- Thermal capacitance,
- Connected to neighboring cells via thermal resistances (RC thermal network).

The heat transfer process is governed by the heat equation.

---

### 2. Numerical Resolution

Two main numerical approaches were implemented:

- **Crank–Nicolson scheme** for transient heat diffusion  
  (stability–accuracy tradeoff between explicit and implicit schemes)

- **Conjugate Gradient method** for solving the linear systems resulting from spatial discretization in steady-state regimes

The implementation focuses on numerical stability, convergence behavior, and computational efficiency.

---

### 3. Validation

The model is conceptually aligned with experimental approaches such as infrared thermography measurements used in CPU thermal studies.

Results are analyzed in terms of:

- Temperature gradients
- Hotspot intensity
- Spatial distribution of heat

---

### 4. Geometric Optimization

The spatial configuration of functional blocks on the die is modified in order to:

- Reduce peak temperatures,
- Minimize thermal gradients,
- Improve heat spreading efficiency.

Different geometric layouts are evaluated through simulation.

---

## Technical Stack

- Python
- NumPy
- Scientific computing tools
- Iterative linear solvers (Conjugate Gradient implementation)

---

## Repository Structure

