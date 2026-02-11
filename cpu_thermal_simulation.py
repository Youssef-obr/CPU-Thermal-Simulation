import random
import copy
import math
from collections import defaultdict


import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

#DIFF 6°C 6000 iter
#DIFF 5  10k, 2.5 iter


# -----------------------------
# Paramètres physiques
# -----------------------------
dx = dy = 1e-3        # m
e = 200e-6            # m
k = 150               # W/m·K
rho_c = 1.66e6        # J/m³·K
h = 2000            # W/m²·K
Rb = 200 #46.3             # K/W
T_amb = 25.0          # °C
T_sub = 25.0          # °C
dt = 0.1              # s
t_end = 100.0           # s

# -----------------------------
# Grille de blocs fonctionnels (10x10) 
# -----------------------------
grid_10x10 = [
    ['IO', 'IO',   'IO',   'IO',   'IO',   'IO',   'IO',   'IO',   'IO',   'IO'],
    ['IO', 'GPU', 'GPU', 'GPU', 'GPU', 'GPU', 'MEM', 'CRYPT', 'CRYPT', 'IO'],
    ['IO', 'GPU', 'GPU', 'GPU', 'GPU', 'GPU', 'MEM', 'CRYPT', 'CRYPT', 'IO'],
    ['IO', 'GPU', 'GPU', 'GPU', 'GPU', 'GPU', 'MEM', 'PM',    'PM',    'IO'],
    ['IO', 'U.S.T','U.S.T','U.S.T','U.S.T','CORE', 'CORE', 'PM',    'PM',    'IO'],
    ['IO', 'U.S.T','U.S.T','U.S.T','U.S.T','CORE', 'CORE', 'AUDIO', 'AUDIO', 'IO'],
    ['IO', 'C.A',  'C.A',  'VIDEO','VIDEO','CORE', 'CORE', 'AUDIO', 'AUDIO', 'IO'],
    ['IO', 'C.A',  'C.A',  'VIDEO','VIDEO','CORE', 'CORE', 'AUDIO', 'AUDIO', 'IO'],
    ['IO', 'C.A',  'C.A',  'VIDEO','VIDEO','CORE', 'CORE', 'AUDIO', 'AUDIO', 'IO'],
    ['IO', 'IO',   'IO',   'IO',   'IO',   'IO',   'IO',   'IO',   'IO',   'IO'],

]

ny, nx = len(grid_10x10), len(grid_10x10[0])
N = ny * nx
Ac = dx * dy
Rl = dx / (k * e * dy)
C = rho_c * dx * dy * e

# -----------------------------
# Dictionnaire des puissances (W) 
# -----------------------------
power_dict = {
    'CORE':   5.2 / 10,
    'GPU':    3 / 15,
    'AUDIO':  0.5 / 8,
    'VIDEO':  0.6 / 6,
    'C.A':    0.4 / 6,
    'U.S.T':  0.3 / 8,
    'CRYPT':  0.3 / 4,
    'PM':     0.2 / 4,
    'MEM':    0.8 / 3,
    'IO':     0.7 / 36  # 28+5 IO blocs
}

# -----------------------------
# Fonctions auxiliaires
# -----------------------------
def idx(i, j): return i * nx + j
def voisins(i, j):
    v = []
    if i > 0: v.append((i - 1, j))
    if i < ny - 1: v.append((i + 1, j))
    if j > 0: v.append((i, j - 1))
    if j < nx - 1: v.append((i, j + 1))
    return v

# -----------------------------
# Assemblage matrices A et B
# -----------------------------
def assemble_matrices():
    A = sp.lil_matrix((N, N))
    B = sp.lil_matrix((N, N))
    for i in range(ny):
        for j in range(nx):
            p = idx(i, j)
            a_diag = C / dt + 0.5 * (h * Ac + 1 / Rb)
            b_diag = C / dt - 0.5 * (h * Ac + 1 / Rb)
            for (x, y) in voisins(i, j):
                a_diag += 0.5 / Rl
                b_diag -= 0.5 / Rl
            A[p, p] = a_diag
            B[p, p] = b_diag
            for (x, y) in voisins(i, j):
                q = idx(x, y)
                A[p, q] = -0.5 / Rl
                B[p, q] = +0.5 / Rl
    return A.tocsr(), B.tocsr()

# -----------------------------
# Simulation
# -----------------------------
def simulate(grid, power_dict):
    A, B = assemble_matrices()
    T = np.full(N, T_amb)

    # Construction du vecteur P
    P = np.array([
        power_dict[grid[i][j]] for i in range(ny) 
        for j in range(nx)
    ])

    n_steps = int(t_end / dt)
    for _ in range(n_steps):
        S = P + np.full(N, h * Ac * T_amb +  T_sub / Rb)
        rhs = B @ T + S
        T, _ = spla.cg(A, rhs, x0=T, atol=1e-8)
    return T.reshape((ny, nx))

# -----------------------------
# Exécution
# -----------------------------
T_result = simulate(grid_10x10, power_dict)

# Affichage
plt.imshow(T_result, cmap='coolwarm', origin='upper')
plt.colorbar(label="Température (°C)")
plt.title("Température finale (grille 10×10)")
plt.xticks(range(nx))
plt.yticks(range(ny))
plt.show()
print(f"Tmax: {T_result.max():.2f} °C et Tmin: {T_result.min():.2f}")
print(f"Tmoy: {T_result.mean():.2f} °C")



def optimize_and_return(grid_10x10, power_dict, n_iter=1000):
    current_grid = [row[:] for row in grid_10x10]
    current_T = simulate(current_grid, power_dict)
    current_Tmax = current_T.max()
    
    best_grid = copy.deepcopy(current_grid)
    best_Tmax = current_Tmax
    best_heatmap = current_T

    # Paramètres optimisés du recuit
    initial_temp = 5.0
    cooling_rate = 0.995

    for k in range(n_iter):
        temp = initial_temp * (cooling_rate ** k)
        
        new_grid = generate_valid_neighbor(current_grid)
        new_T = simulate(new_grid, power_dict)
        new_Tmax = new_T.max()

        if new_Tmax < current_Tmax or random.random() < math.exp((current_Tmax - new_Tmax)/max(temp, 1e-6)):
            current_grid, current_Tmax = new_grid, new_Tmax
            if new_Tmax < best_Tmax:
                best_grid, best_Tmax, best_heatmap = copy.deepcopy(new_grid), new_Tmax, new_T

    # Affichage des résultats
    print(f"Tmax initiale: {simulate(grid_10x10, power_dict).max():.2f}°C")
    print(f"Tmax optimisée: {best_Tmax:.2f}°C\n")

    # Heatmap finale
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(simulate(grid_10x10, power_dict), cmap='coolwarm', origin='upper')
    plt.colorbar(label="Température (°C)")
    plt.title("Configuration Initiale\nTmax = {current_Tmax:.2f}°C")
    
    plt.subplot(1, 2, 2)
    plt.imshow(best_heatmap, cmap='coolwarm', origin='upper')
    plt.colorbar(label="Température (°C)")
    plt.title(f"Configuration Optimisée\nTmax = {best_Tmax:.2f}°C")
    
    plt.tight_layout()
    plt.show()

    return best_grid, best_heatmap, best_Tmax, best_heatmap.min()

def generate_valid_neighbor(grid):
    """Échange deux blocs en conservant la connectivité des clusters"""
    new_grid = [row[:] for row in grid]
    attempts = 0
    
    while attempts < 100:
        # Trouve tous les blocs échangeables
        swap_candidates = [(i,j) for i in range(1,9) for j in range(1,9) if grid[i][j] != 'IO']
        
        if len(swap_candidates) < 2:
            return new_grid
        # Choisit deux blocs différents
        (i1,j1), (i2,j2) = random.sample(swap_candidates, 2)
        block1, block2 = grid[i1][j1], grid[i2][j2]
        
        # Effectue l'échange
        new_grid[i1][j1], new_grid[i2][j2] = block2, block1
        
        # Vérifie la connectivité des clusters concernés
        if (is_cluster_connected(new_grid, block1) and 
            is_cluster_connected(new_grid, block2)):
            return new_grid
        else:
            # Annule l'échange si la connectivité est rompue
            new_grid[i1][j1], new_grid[i2][j2] = block1, block2
            attempts += 1
    
    return new_grid

def is_cluster_connected(grid, block_type):
    """Vérifie si tous les blocs du même type sont connectés"""
    if block_type == 'IO':
        return True
        
    positions = [(i,j) for i in range(10) for j in range(10) if grid[i][j] == block_type]
    if not positions:
        return True
        
    # Trouve le cluster connecté
    cluster = find_connected_components(grid, positions[0][0], positions[0][1])
    return len(cluster) == len(positions)

def find_connected_components(grid, i, j):
    """Trouve toutes les cellules connectées du même type """
    cluster = []
    queue = [(i,j)]
    visited = set()
    block_type = grid[i][j]
    
    while queue:
        x, y = queue.pop(0)
        if (x,y) not in visited and 0 <= x < 10 and 0 <= y < 10 and grid[x][y] == block_type:
            visited.add((x,y))
            cluster.append((x,y))
            queue.extend([(x+1,y), (x-1,y), (x,y+1), (x,y-1)])
    
    return cluster

# Exemple d'utilisation
optimized_grid, heatmap, Tmax, Tmin = optimize_and_return(grid_10x10, power_dict, n_iter=200)

# Affichage texte de la grille optimisée
print("Grille optimisée :")
for row in optimized_grid:
    print("  ".join(f"{cell:6}" for cell in row))