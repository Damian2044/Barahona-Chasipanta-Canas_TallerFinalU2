# Informe Tecnico - Pipeline Hibrido para Procesamiento Masivo de Imagenes

## Dataset seleccionado

- Nombre: Corn Leaf Disease
- Kaggle: https://www.kaggle.com/datasets/ndisan/corn-leaf-disease
- Carpeta local del proyecto: `data/corn-leaf-disease`

El script `src/proyecto/datos/descargarDataset.py` descarga con KaggleHub y copia el dataset dentro de `data/` si aun no existe. El tamano se comprueba con `PYTHONPATH=src python -m proyecto.metricas.verificarDataset`; en las metricas guardadas anteriormente se registraron 4000 imagenes y cerca de 5.11 GB.

## Arquitectura

```text
KaggleHub/data -> Dask DataFrame -> Multiprocessing/OpenCV -> NumPy/RAW -> CUDA Laplaciano -> Integracion -> Streamlit
```

### Fase 1: Dask DataFrame

`src/proyecto/fases/dask.py` usa Dask DataFrame con particiones lazy creadas desde rutas de imagenes. La fase limpia extensiones, valida tamanos, tipa columnas y ejecuta un `groupby` por clase antes de consolidar resultados.

Salidas:

- `salidas/tablaRutas.csv`
- `metricas/metricasDask.csv`
- `metricas/metricasDaskClases.csv`

### Fase 2: Multiprocessing + OpenCV

`src/proyecto/fases/multiprocessingCpu.py` usa `multiprocessing.Pool` y chunks. Por defecto usa todos los CPU disponibles. Cada imagen se lee con OpenCV, se convierte a gris, se redimensiona a `512x512`, se normaliza para NumPy y se exporta tambien como RAW `uint8` para CUDA.

Metricas por imagen:

- dimensiones originales;
- canales originales;
- bytes originales;
- dimensiones y bytes luego del resize;
- media, desviacion, minimo, maximo, mediana y percentiles;
- tiempo por imagen, chunk, procesos usados y tiempo real de fase.

Salidas:

- `salidas/npyCpu/`
- `salidas/rawCuda/`
- `metricas/metricasCpu.csv`
- `metricas/metricasCpuClases.csv`
- `metricas/metricasPrepararCuda.csv`
- `salidas/paquetes/paqueteCudaColab.zip`

### Fase 3: CUDA

El notebook `src/proyecto/fases/cuda/Barahona,Chasipanta,Cañas_Cuda.ipynb` esta preparado para Google Colab. Descomprime `paqueteCudaColab.zip`, ejecuta un kernel `%%cuda` y genera `resultadosCuda.zip` en Colab.

El kernel `filtroLaplacianoMemoriaCompartida` aplica un filtro laplaciano usando memoria compartida por bloque. La configuracion de imagen e hilos esta definida arriba:

- `TAMANO_IMAGEN = 512`
- `HILOS_X = 16`
- `HILOS_Y = 16`

Salidas desde Colab:

- `salidas/cudaBin/`
- `metricas/metricasCudaColab.csv`
- `salidas/paquetes/resultadosCuda.zip`

### Fase 4: Integracion y Streamlit

`src/proyecto/fases/integrarCuda.py` descomprime `salidas/paquetes/resultadosCuda.zip`, convierte los binarios CUDA a `.npy` y `.png`, y calcula metricas del resultado laplaciano.

`appStreamlit.py` muestra la evidencia final con Plotly: tiempos reales, tamanos, dimensiones, distribuciones por clase, metricas CUDA e imagenes procesadas.

## Comandos

```bash
python main.py
python main.py --etapa integrar-cuda --zip-resultados salidas/paquetes/resultadosCuda.zip
python main.py --etapa resumen
streamlit run appStreamlit.py
```
