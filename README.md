# Delivery Dynamics – Optimización de rutas de drones

Este proyecto compara 5 algoritmos para la optimización de rutas de entrega con drones, usando una matriz de distancias entre ciudades.

## Requisitos

- Python 3.8 o superior
- pip

## Instalación

1. **Clona el repositorio** (o descarga los archivos):

   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd delivery_dynamics
   ```

2. **Instala las dependencias**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Asegúrate de tener la matriz de distancias**  
   El archivo `matriz_distancias_suchitepequez.csv` debe estar en la raíz del proyecto, en este momento ya se incluye.

## Uso

Ejecuta el script principal con:

```bash
python main.py --matrix matriz_distancias_suchitepequez.csv --generations 500 --population 150 --seed 42
```

Parámetros principales:

- `--matrix`: Ruta al archivo CSV de la matriz de distancias.
- `--generations` (`-g`): Número de generaciones (default: 500).
- `--population` (`-p`): Tamaño de la población para los algoritmos evolutivos (default: 150).
- `--seed`: Semilla aleatoria (opcional).

## Salida

- Progreso y resultados de cada algoritmo en consola.
- Gráficas de evolución del fitness y rutas óptimas encontradas.

## Notas

- El punto de inicio y retorno es siempre "Mazatenango".
- La capacidad de entrega por viaje está definida en el código (`DELIVERY_CAPACITY`).

## Informe Técnico

Este proyecto implementa y compara cinco algoritmos ligeros para la optimización de rutas de entrega por dron en el departamento de Suchitepéquez, Guatemala. El dron parte y regresa a la cabecera departamental, Mazatenango, realizando entregas con una capacidad limitada antes de retornar al punto de partida.

### Algoritmos implementados

1. **GA generacional elitista (GA)**  
   Algoritmo genético clásico con elitismo. En cada generación, una fracción de los mejores individuos se preserva, mientras que el resto de la población se genera mediante selección por torneo, cruce ordenado y mutación por intercambio.

2. **Steady-State GA (SSGA)**  
   Variante del GA donde en cada iteración se reemplaza únicamente el peor individuo si un nuevo individuo mejor lo supera. Esto permite una evolución más estable y continua sin reemplazar toda la población.

3. **Greedy + 2-Opt (G2O)**  
   Algoritmo heurístico rápido. Se construye una ruta inicial usando el vecino más cercano y luego se refina localmente mediante la técnica de optimización 2‑Opt, que intercambia segmentos para reducir la distancia.

4. **Simulated Annealing (SA)**  
   Metaheurística inspirada en el enfriamiento del metal. A partir de una solución aleatoria, se aceptan empeoramientos temporales en la búsqueda con una probabilidad decreciente, lo que ayuda a escapar de óptimos locales.

5. **Random-Restart Hill Climb (RRHC)**  
   Realiza múltiples búsquedas locales desde rutas aleatorias. Cada búsqueda aplica Hill Climb hasta alcanzar un óptimo local, y se conserva la mejor solución global encontrada.

### Componentes de los algoritmos genéticos

Componentes genéricos usados en las variantes de algoritmos genéticos implementadas:

- **Representación del individuo (gen)** (`create_individual`): Cada individuo representa una ruta, codificada como una permutación de las ciudades a visitar (excluyendo Mazatenango, que es el punto de partida y retorno). Esta representación es adecuada para problemas de ruteo como el TSP.

- **Función de fitness** (`total_distance`): Se evalúa la distancia total recorrida en la ruta, tomando en cuenta la capacidad de entregas. Cuanto menor la distancia, mejor el fitness.

- **Población inicial** (`create_individual` dentro de una lista por comprensión): Se genera aleatoriamente creando múltiples permutaciones de las ciudades. Esto garantiza diversidad desde el comienzo.

- **Selección** (`tournament`): Se usa selección por torneo. Se escogen `k` individuos aleatoriamente y se selecciona el de mejor fitness.

- **Cruce (crossover)** (`ordered_crossover`): Se emplea el operador Ordered Crossover (OX), que garantiza que los hijos sean rutas válidas sin duplicar ciudades.

- **Mutación** (`swap_mutation`): Se aplica mutación por intercambio (swap), intercambiando pares de ciudades con cierta probabilidad. Esto introduce pequeñas variaciones para evitar estancamientos.

- **Elitismo** (dentro de `ga_elitist`): Un porcentaje fijo de los mejores individuos se conserva directamente a la siguiente generación, preservando soluciones óptimas.

### Consideraciones técnicas

- La función de evaluación es la distancia total recorrida, considerando la capacidad máxima de entregas (`DELIVERY_CAPACITY`) antes de regresar a Mazatenango.
- Todos los algoritmos registran su evolución de fitness y se grafican sobre la misma escala de generaciones.
- Se incluyen visualizaciones de las rutas óptimas encontradas en formato gráfico para su análisis comparativo.


