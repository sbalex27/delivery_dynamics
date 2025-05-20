# drone_route_ga.py
"""
Proyecto: Optimización de Rutas de Drones con Algoritmos Genéticos
Departamento de Suchitepéquez, Guatemala 18 puntos de entrega, punto de partida filial Mazatenango

Ejecución rápida
---------------
1. Instala dependencias (idealmente en un entorno virtual):
   pip install -r requirements.txt

2. Coloca el archivo CSV de distancias (matriz_distancias_suchitepequez.csv) en el mismo directorio.

3. Ejecuta:
   python drone_route_ga.py --matrix matriz_distancias_suchitepequez.csv --generations 500 --population 150 --seed 42

   Se abrirán dos ventanas: la gráfica de evolución del fitness y un grafo con la mejor ruta encontrada.

Dependencias
------------
- pandas
- numpy
- matplotlib
- networkx

Descripción rápida
------------------
- **Cromosoma**: Permutación de los 18 municipios a visitar.
- **Fitness**: Distancia total (km) recorrida por un único dron con capacidad de 5 paquetes.  
  Cada vez que se alcanzan 5 entregas el dron regresa a la base a recargar.
- **Selección**: Torneo (k=3).
- **Cruzamiento**: Ordered Crossover (OX).
- **Mutación**: Intercambio (swap) con probabilidad p=0.05.
- **Élite**: 5 % mejores rutas pasan intactas a la siguiente generación.
- **Visualización**: matplotlib (línea de fitness) + networkx (grafo de la ruta).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# ----------------------------- Modelo de Problema ---------------------------- #

START_CITY = "Mazatenango"
DELIVERY_CAPACITY = 5  # máx. paquetes por viaje

# ----------------------------- Utilidades GA --------------------------------- #

def total_distance(route: List[str], dist: pd.DataFrame, start: str = START_CITY, cap: int = DELIVERY_CAPACITY) -> float:
    """Calcula la distancia total, obligando retorno cada *cap* entregas."""
    total = 0.0
    current, load = start, 0
    for city in route:
        total += dist.loc[current, city]
        current, load = city, load + 1
        if load == cap:  # regreso para recargar
            total += dist.loc[current, start]
            current, load = start, 0
    if current != start:  # regreso final
        total += dist.loc[current, start]
    return total


def create_individual(cities: List[str]) -> List[str]:
    ind = cities.copy()
    random.shuffle(ind)
    return ind


def ordered_crossover(p1: List[str], p2: List[str]) -> List[str]:
    N = len(p1)
    a, b = sorted(random.sample(range(N), 2))
    child = [None] * N
    child[a : b + 1] = p1[a : b + 1]
    pos = (b + 1) % N
    for gene in p2:
        if gene not in child:
            child[pos] = gene
            pos = (pos + 1) % N
    return child


def swap_mutation(route: List[str], p: float = 0.05) -> List[str]:
    r = route.copy()
    for i in range(len(r)):
        if random.random() < p:
            j = random.randrange(len(r))
            r[i], r[j] = r[j], r[i]
    return r


# ----------------------------- Núcleo GA ------------------------------------- #

def genetic_algorithm(
    dist: pd.DataFrame,
    start_city: str = START_CITY,
    pop_size: int = 150,
    generations: int = 500,
    elite_size: int = 5,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.05,
    tournament_k: int = 3,
    seed: int | None = None,
) -> Tuple[Dict, List[float]]:
    """Evoluciona rutas y devuelve (mejor_individuo, historial_fitness)."""

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    delivery_cities = [c for c in dist.index if c != start_city]

    def fitness(route: List[str]) -> float:
        return total_distance(route, dist, start=start_city, cap=DELIVERY_CAPACITY)

    population = [{"route": create_individual(delivery_cities)} for _ in range(pop_size)]
    for ind in population:
        ind["fitness"] = fitness(ind["route"])

    history = []
    for _ in range(generations):
        population.sort(key=lambda ind: ind["fitness"])  # mejor al inicio
        history.append(population[0]["fitness"])

        # ---------------- Nuevo pool ---------------- #
        new_pop: List[Dict] = population[:elite_size]  # élite
        while len(new_pop) < pop_size:
            if random.random() < crossover_rate:
                p1 = tournament(population, tournament_k)
                p2 = tournament(population, tournament_k)
                child_route = ordered_crossover(p1["route"], p2["route"])
            else:
                parent = tournament(population, tournament_k)
                child_route = parent["route"].copy()
            child_route = swap_mutation(child_route, mutation_rate)
            new_pop.append({"route": child_route, "fitness": fitness(child_route)})
        population = new_pop

    population.sort(key=lambda ind: ind["fitness"])
    return population[0], history


def tournament(pop: List[Dict], k: int) -> Dict:
    return min(random.sample(pop, k), key=lambda ind: ind["fitness"])

# ----------------------------- Visualización --------------------------------- #

def plot_fitness(history: List[float]):
    plt.figure(figsize=(9, 5))
    plt.plot(history)
    plt.title("Evolución del mejor fitness")
    plt.xlabel("Generación")
    plt.ylabel("Distancia (km)")
    plt.tight_layout()
    plt.show()


def plot_route(route: List[str], start_city: str = START_CITY):
    cities = [start_city] + route + [start_city]
    G = nx.Graph()
    G.add_nodes_from(cities)
    G.add_edges_from([(cities[i], cities[i + 1]) for i in range(len(cities) - 1)])

    pos = nx.spring_layout(G, seed=3)  # disposición estética reproducible
    nx.draw_networkx_nodes(G, pos, node_color="#8ecae6", node_size=800)
    nx.draw_networkx_edges(G, pos, edge_color="#ffb703", width=2)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title("Mejor ruta encontrada (orden de visita y retornos)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# ----------------------------- CLI ------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Optimización de Ruta de Dron con GA")
    p.add_argument("--matrix", type=Path, default="matriz_distancias_suchitepequez.csv", help="CSV con la matriz de distancias")
    p.add_argument("--generations", "-g", type=int, default=500)
    p.add_argument("--population", "-p", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.matrix, index_col=0)

    best, history = genetic_algorithm(
        df,
        pop_size=args.population,
        generations=args.generations,
        seed=args.seed,
    )

    print("\n📦  Ruta óptima (orden de entrega):")
    print(" -> ".join([START_CITY] + best["route"] + [START_CITY]))
    print(f"\n🚀  Distancia total: {best['fitness']:.2f} km")

    # Visualizaciones
    plot_fitness(history)
    plot_route(best["route"], start_city=START_CITY)


if __name__ == "__main__":
    main()
