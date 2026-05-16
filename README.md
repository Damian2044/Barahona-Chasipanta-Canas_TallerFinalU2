# Taller Final Unidad 2 - Pipeline Hibrido de Imagenes

Arquitectura implementada:

```text
KaggleHub/data -> Dask DataFrame -> Multiprocessing/OpenCV -> NumPy/RAW -> CUDA Laplaciano -> Integracion -> Streamlit
```

Dataset:

<https://www.kaggle.com/datasets/ndisan/corn-leaf-disease>

## Instalacion

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Ejecucion local por defecto

```bash
python main.py
```

Por defecto:

- descarga el dataset en `data/corn-leaf-disease` si no existe;
- limpia salidas anteriores;
- ejecuta Dask DataFrame;
- procesa imagenes con `multiprocessing.Pool` usando todos los CPU disponibles;
- redimensiona a `512x512` con OpenCV;
- guarda `.npy`, RAW `uint8` y metricas;
- crea `salidas/paquetes/paqueteCudaColab.zip` listo para subir a Colab;
- genera `metricas/metricasResumen.csv`.

Para conservar resultados existentes:

```bash
python main.py --no-borrar
```

## Ejecutar etapas sueltas

```bash
python main.py --etapa descarga
python main.py --etapa dask
python main.py --etapa cpu --procesos 8
python main.py --etapa zip-cuda
python main.py --etapa integrar-cuda --zip-resultados salidas/paquetes/resultadosCuda.zip
python main.py --etapa resumen
```

## CUDA en Colab

1. Sube `salidas/paquetes/paqueteCudaColab.zip` a Colab.
2. Abre `src/proyecto/fases/cuda/Barahona,Chasipanta,Cañas_Cuda.ipynb`.
3. Ejecuta las celdas con runtime GPU.
4. Descarga `resultadosCuda.zip`.
5. Guarda `resultadosCuda.zip` como `salidas/paquetes/resultadosCuda.zip`.
6. Ejecuta:

```bash
python main.py --etapa integrar-cuda --zip-resultados salidas/paquetes/resultadosCuda.zip
```

El kernel CUDA aplica un filtro laplaciano con memoria compartida sobre imagenes `512x512`.

## Streamlit

```bash
streamlit run appStreamlit.py
```

La interfaz muestra metricas de Dask, dimensiones originales, tamano original, tamano despues del resize, tiempos reales de fase, metricas CUDA y resultados visuales.
