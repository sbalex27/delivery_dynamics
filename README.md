# Delivery Dynamics – Optimización de rutas de drones

Este proyecto compara 5 algoritmos ligeros para la optimización de rutas de entrega con drones, usando una matriz de distancias entre ciudades.

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

---
