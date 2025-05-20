# drone_route_ga.py
"""
Optimización de rutas de drones - Comparativa de 5 algoritmos ligeros
====================================================================
1. **GA generacional elitista** (GA)          - referencia.
2. **Steady-State GA** (SSGA)                 - reemplazo parcial.
3. **Greedy + 2-Opt** (G2O)                  - heurístico instantáneo.
4. **Simulated Annealing** (SA)               - enfriamiento simple.
5. **Random-Restart Hill Climb** (RRHC)       - varias búsquedas locales.

Todos muestran progreso en consola y se grafican en la misma escala de generaciones.

Uso
---
```bash
python drone_route_ga.py --matrix matriz_distancias_suchitepequez.csv \
                         --generations 500 --population 150 --seed 42
```
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

START_CITY = "Mazatenango"  # Ciudad de inicio y fin de la ruta
DELIVERY_CAPACITY = 5       # Capacidad máxima de entregas por viaje
ELITE_FRAC = 0.05           # Fracción de élite en el GA generacional

# --------------------------------------------------------------------------- #
# Utilidades generales                                                         #
# --------------------------------------------------------------------------- #

def total_distance(route: List[str], dist: pd.DataFrame, start: str = START_CITY, cap: int = DELIVERY_CAPACITY) -> float:
    """
    Calcula la distancia total de una ruta considerando la capacidad de entregas.
    
    Args:
        route (List[str]): Lista de ciudades a visitar en orden.
        dist (pd.DataFrame): Matriz de distancias entre ciudades.
        start (str): Ciudad de inicio y fin.
        cap (int): Capacidad máxima de entregas antes de regresar al inicio.
    
    Returns:
        float: Distancia total recorrida.
    """
    total, current, load = 0.0, start, 0
    for city in route:
        total += dist.loc[current, city]
        current, load = city, load + 1
        if load == cap:
            total += dist.loc[current, start]
            current, load = start, 0
    if current != start:
        total += dist.loc[current, start]
    return total


def create_individual(cities: List[str]) -> List[str]:
    """
    Crea un individuo (ruta) aleatorio para la población inicial.
    
    Args:
        cities (List[str]): Lista de ciudades (sin incluir la ciudad de inicio).
    
    Returns:
        List[str]: Ruta aleatoria.
    """
    ind = cities.copy()
    random.shuffle(ind)
    return ind

# --------------------------------------------------------------------------- #
# Operadores GA                                                                #
# --------------------------------------------------------------------------- #

def tournament(pop: List[Dict], k: int = 3) -> Dict:
    """
    Selección por torneo: elige el mejor individuo entre k seleccionados al azar.
    
    Args:
        pop (List[Dict]): Población de individuos.
        k (int): Tamaño del torneo.
    
    Returns:
        Dict: Individuo ganador del torneo.
    """
    return min(random.sample(pop, k), key=lambda i: i["fitness"])


def ordered_crossover(p1: List[str], p2: List[str]) -> List[str]:
    """
    Cruce ordenado (OX): combina dos rutas para crear un hijo válido.
    
    Args:
        p1 (List[str]): Primer padre.
        p2 (List[str]): Segundo padre.
    
    Returns:
        List[str]: Ruta hija resultante del cruce.
    """
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [None] * n
    child[a : b + 1] = p1[a : b + 1]
    pos = (b + 1) % n
    for gene in p2:
        if gene not in child:
            child[pos] = gene
            pos = (pos + 1) % n
    return child


def swap_mutation(route: List[str], p: float = 0.05) -> List[str]:
    """
    Mutación por intercambio: intercambia aleatoriamente pares de ciudades.
    
    Args:
        route (List[str]): Ruta a mutar.
        p (float): Probabilidad de mutación por gen.
    
    Returns:
        List[str]: Ruta mutada.
    """
    r = route.copy()
    for i in range(len(r)):
        if random.random() < p:
            j = random.randrange(len(r))
            r[i], r[j] = r[j], r[i]
    return r

# --------------------------------------------------------------------------- #
# 2‑Opt y heurística Greedy                                                    #
# --------------------------------------------------------------------------- #

def two_opt(route: List[str], dist: pd.DataFrame) -> List[str]:
    """
    Optimización local 2-Opt: mejora la ruta invirtiendo segmentos.
    
    Args:
        route (List[str]): Ruta inicial.
        dist (pd.DataFrame): Matriz de distancias.
    
    Returns:
        List[str]: Ruta optimizada localmente.
    """
    improved, best = True, route
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best)):
                if j - i == 1:
                    continue
                new_route = best[:i] + best[i:j][::-1] + best[j:]
                if total_distance(new_route, dist) < total_distance(best, dist):
                    best = new_route
                    improved = True
    return best


def nearest_neighbour(cities: List[str], dist: pd.DataFrame) -> List[str]:
    """
    Construye una ruta usando el heurístico del vecino más cercano.
    
    Args:
        cities (List[str]): Lista de ciudades a visitar.
        dist (pd.DataFrame): Matriz de distancias.
    
    Returns:
        List[str]: Ruta generada por el heurístico.
    """
    unvisited = set(cities)
    route, current = [], START_CITY
    while unvisited:
        nxt = min(unvisited, key=lambda c: dist.loc[current, c])
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return route

# --------------------------------------------------------------------------- #
# Algoritmo 1 – GA generacional elitista                                       #
# --------------------------------------------------------------------------- #

def ga_elitist(dist: pd.DataFrame, *, pop_size: int, generations: int, mut: float, cr: float, seed=None):
    """
    Algoritmo genético generacional con elitismo.
    
    Args:
        dist (pd.DataFrame): Matriz de distancias.
        pop_size (int): Tamaño de la población.
        generations (int): Número de generaciones.
        mut (float): Probabilidad de mutación.
        cr (float): Probabilidad de cruce.
        seed (opcional): Semilla para reproducibilidad.
    
    Returns:
        Tuple[Dict, List[float]]: Mejor individuo y evolución del fitness.
    """
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    cities = [c for c in dist.index if c != START_CITY]
    fitness = lambda r: total_distance(r, dist)
    elite = max(1, int(pop_size * ELITE_FRAC))
    pop = [{"route": create_individual(cities)} for _ in range(pop_size)]
    for ind in pop:
        ind["fitness"] = fitness(ind["route"])
    hist = []
    for g in range(generations):
        pop.sort(key=lambda i: i["fitness"])
        best_fit = pop[0]["fitness"]
        hist.append(best_fit)
        sys.stdout.write(f"GA  Gen {g+1}/{generations} – {best_fit:.2f} km\r"); sys.stdout.flush()
        new_pop = pop[:elite]
        while len(new_pop) < pop_size:
            route = tournament(pop)["route"].copy() if random.random() > cr else ordered_crossover(tournament(pop)["route"], tournament(pop)["route"])
            route = swap_mutation(route, mut)
            new_pop.append({"route": route, "fitness": fitness(route)})
        pop = new_pop
    print()
    pop.sort(key=lambda i: i["fitness"])
    return pop[0], hist

# --------------------------------------------------------------------------- #
# Algoritmo 2 – Steady‑State GA                                               #
# --------------------------------------------------------------------------- #

def ss_ga(dist: pd.DataFrame, *, pop_size: int, generations: int, mut: float, cr: float, seed=None):
    """
    Algoritmo genético steady-state (reemplazo parcial).
    
    Args:
        dist (pd.DataFrame): Matriz de distancias.
        pop_size (int): Tamaño de la población.
        generations (int): Número de generaciones equivalentes.
        mut (float): Probabilidad de mutación.
        cr (float): Probabilidad de cruce.
        seed (opcional): Semilla para reproducibilidad.
    
    Returns:
        Tuple[Dict, List[float]]: Mejor individuo y evolución del fitness.
    """
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    cities = [c for c in dist.index if c != START_CITY]
    fitness = lambda r: total_distance(r, dist)
    pop = [{"route": create_individual(cities)} for _ in range(pop_size)]
    for ind in pop:
        ind["fitness"] = fitness(ind["route"])
    hist, evals = [], generations * pop_size
    for i in range(evals):
        child = tournament(pop)["route"].copy() if random.random() > cr else ordered_crossover(tournament(pop)["route"], tournament(pop)["route"])
        child = swap_mutation(child, mut)
        child_fit = fitness(child)
        worst = max(pop, key=lambda x: x["fitness"])
        if child_fit < worst["fitness"]:
            pop.remove(worst); pop.append({"route": child, "fitness": child_fit})
        if (i + 1) % pop_size == 0:
            gen = (i + 1) // pop_size
            best_fit = min(pop, key=lambda x: x["fitness"])["fitness"]
            hist.append(best_fit)
            sys.stdout.write(f"SSGA Gen {gen}/{generations} – {best_fit:.2f} km\r"); sys.stdout.flush()
    print()
    best = min(pop, key=lambda x: x["fitness"])
    return best, hist

# --------------------------------------------------------------------------- #
# Algoritmo 3 – Greedy + 2‑Opt                                                #
# --------------------------------------------------------------------------- #

def greedy_2opt(dist: pd.DataFrame, *, generations: int):
    """
    Algoritmo heurístico: vecino más cercano seguido de optimización 2-Opt.
    
    Args:
        dist (pd.DataFrame): Matriz de distancias.
        generations (int): Número de generaciones equivalentes (para graficar).
    
    Returns:
        Tuple[Dict, List[float]]: Mejor ruta y evolución del fitness.
    """
    cities = [c for c in dist.index if c != START_CITY]
    print("G2O: Vecino más cercano…", end=" ")
    route = nearest_neighbour(cities, dist)
    print("2‑Opt…", end=" ")
    route = two_opt(route, dist)
    fit = total_distance(route, dist)
    print(f"Listo {fit:.2f} km")
    hist = [fit] * generations
    return {"route": route, "fitness": fit}, hist

# --------------------------------------------------------------------------- #
# Algoritmo 4 – Simulated Annealing                                           #
# --------------------------------------------------------------------------- #

def simulated_annealing(dist: pd.DataFrame, *, generations: int, steps_per_gen: int = 100, t0: float = 1000.0, alpha: float = 0.995, seed=None):
    """
    Recocido simulado (Simulated Annealing) para optimización de rutas.
    
    Args:
        dist (pd.DataFrame): Matriz de distancias.
        generations (int): Número de generaciones equivalentes.
        steps_per_gen (int): Pasos por generación.
        t0 (float): Temperatura inicial.
        alpha (float): Factor de enfriamiento.
        seed (opcional): Semilla para reproducibilidad.
    
    Returns:
        Tuple[Dict, List[float]]: Mejor ruta y evolución del fitness.
    """
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    cities = [c for c in dist.index if c != START_CITY]
    current = create_individual(cities)
    current_fit = total_distance(current, dist)
    best, best_fit = current, current_fit
    T = t0
    hist = []
    total_steps = generations * steps_per_gen
    for step in range(total_steps):
        # vecindario: swap aleatorio
        i, j = random.sample(range(len(current)), 2)
        candidate = current.copy(); candidate[i], candidate[j] = candidate[j], candidate[i]
        cand_fit = total_distance(candidate, dist)
        if cand_fit < current_fit or random.random() < math.exp((current_fit - cand_fit) / T):
            current, current_fit = candidate, cand_fit
            if cand_fit < best_fit:
                best, best_fit = candidate, cand_fit
        T *= alpha
        if (step + 1) % steps_per_gen == 0:
            gen = (step + 1) // steps_per_gen
            hist.append(best_fit)
            sys.stdout.write(f"SA  Gen {gen}/{generations} – {best_fit:.2f} km\r"); sys.stdout.flush()
    print()
    return {"route": best, "fitness": best_fit}, hist

# --------------------------------------------------------------------------- #
# Algoritmo 5 – Random‑Restart Hill Climb                                     #
# --------------------------------------------------------------------------- #

def hill_climb(route: List[str], dist: pd.DataFrame):
    """
    Búsqueda local Hill Climb: mejora la ruta por swaps hasta estancarse.
    
    Args:
        route (List[str]): Ruta inicial.
        dist (pd.DataFrame): Matriz de distancias.
    
    Returns:
        List[str]: Ruta optimizada localmente.
    """
    improved = True
    best = route
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                new = best.copy(); new[i], new[j] = new[j], new[i]
                if total_distance(new, dist) < total_distance(best, dist):
                    best = new; improved = True
    return best


def rr_hill_climb(dist: pd.DataFrame, *, generations: int, restarts: int = 50, seed=None):
    """
    Búsqueda local con reinicios aleatorios (Random-Restart Hill Climb).
    
    Args:
        dist (pd.DataFrame): Matriz de distancias.
        generations (int): Número de generaciones equivalentes (para graficar).
        restarts (int): Número de reinicios aleatorios.
        seed (opcional): Semilla para reproducibilidad.
    
    Returns:
        Tuple[Dict, List[float]]: Mejor ruta y evolución del fitness.
    """
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    cities = [c for c in dist.index if c != START_CITY]
    best_route = None
    best_fit = float("inf")
    for r in range(1, restarts + 1):
        route = create_individual(cities)
        route = hill_climb(route, dist)
        fit = total_distance(route, dist)
        if fit < best_fit:
            best_route, best_fit = route, fit
        sys.stdout.write(f"RRHC Restart {r}/{restarts} – Mejor global: {best_fit:.2f} km\r"); sys.stdout.flush()
    print()
    hist = [best_fit] * generations
    return {"route": best_route, "fitness": best_fit}, hist

# --------------------------------------------------------------------------- #
# Visualización                                                               #
# --------------------------------------------------------------------------- #

def plot_histories(histories: List[List[float]], labels: List[str]):
    """
    Grafica la evolución del fitness para cada algoritmo.
    
    Args:
        histories (List[List[float]]): Lista de historiales de fitness.
        labels (List[str]): Etiquetas para cada algoritmo.
    """
    plt.figure(figsize=(9, 5))
    for h, lbl in zip(histories, labels):
        plt.plot(h, label=lbl)
    plt.legend(); plt.title("Evolución del fitness"); plt.xlabel("Generación equivalente"); plt.ylabel("Distancia (km)"); plt.tight_layout(); plt.show()


def plot_route(route: List[str], title: str):
    """
    Grafica la ruta óptima encontrada sobre un grafo circular.
    
    Args:
        route (List[str]): Ruta a graficar.
        title (str): Título del gráfico.
    """
    cities = [START_CITY] + route + [START_CITY]
    G = nx.DiGraph(); G.add_nodes_from(cities); G.add_edges_from([(cities[i], cities[i + 1]) for i in range(len(cities) - 1)])
    pos = {START_CITY: (0, 0)}; angle = 2 * math.pi / (len(cities) - 1); radius = 5
    for idx, city in enumerate(cities[1:-1], 1):
        pos[city] = (radius * math.cos(angle * idx), radius * math.sin(angle * idx))
    plt.figure(figsize=(9, 9)); nx.draw_networkx(G, pos, arrows=True, node_size=700, node_color="#8ecae6", edge_color="#ffb703"); plt.title(title); plt.axis("off"); plt.tight_layout(); plt.show()

# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def parse_args():
    """
    Parsea los argumentos de línea de comandos para la ejecución del script.
    
    Returns:
        argparse.Namespace: Argumentos parseados.
    """
    p = argparse.ArgumentParser("Comparativa: GA, SSGA, G2O, SA, RRHC")
    p.add_argument("--matrix", type=Path, default="matriz_distancias_suchitepequez.csv")
    p.add_argument("--generations", "-g", type=int, default=500, help="Generaciones de referencia")
    p.add_argument("--population", "-p", type=int, default=150)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()

# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    """
    Función principal: ejecuta la comparativa de algoritmos y muestra resultados.
    """
    args = parse_args(); dist = pd.read_csv(args.matrix, index_col=0)

    best_ga, hist_ga = ga_elitist(dist, pop_size=args.population, generations=args.generations, mut=0.05, cr=0.8, seed=args.seed)
    best_ssga, hist_ssga = ss_ga(dist, pop_size=args.population, generations=args.generations, mut=0.05, cr=0.8, seed=args.seed)
    best_g2o, hist_g2o = greedy_2opt(dist, generations=args.generations)
    best_sa, hist_sa = simulated_annealing(dist, generations=args.generations, seed=args.seed)
    best_rrhc, hist_rrhc = rr_hill_climb(dist, generations=args.generations, restarts=50, seed=args.seed)

    print("--- Resultados finales (distancia km) ---")
    print(f"GA     : {best_ga['fitness']:.2f}")
    print(f"SSGA   : {best_ssga['fitness']:.2f}")
    print(f"G2O    : {best_g2o['fitness']:.2f}")
    print(f"SA     : {best_sa['fitness']:.2f}")
    print(f"RRHC   : {best_rrhc['fitness']:.2f}")

    plot_histories([
        hist_ga, hist_ssga, hist_g2o, hist_sa, hist_rrhc
    ], ["GA", "SSGA", "G2O", "SA", "RRHC"])

    plot_route(best_ga["route"], "Ruta óptima GA")
    plot_route(best_ssga["route"], "Ruta óptima SSGA")
    plot_route(best_g2o["route"], "Ruta G2O")
    plot_route(best_sa["route"], "Ruta SA")
    plot_route(best_rrhc["route"], "Ruta RRHC")


if __name__ == "__main__":
    main()
