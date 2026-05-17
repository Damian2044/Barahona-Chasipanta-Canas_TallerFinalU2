# Pasos completos

## 1. Entorno local

```bash
cd C:\Users\aliBa\Desktop\Barahona-Chasipanta-Canas_TallerFinalU2
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Descargar y descomprimir dataset

```bash
python main.py --etapa descarga
```

Salida:

```text
src/data/corn-leaf-disease/
```

## 3. Procesamiento local

```bash
python main.py --no-borrar
```

Fases ejecutadas:

1. Dask crea la tabla de rutas y metricas del dataset.
2. Multiprocessing CPU procesa las imagenes en paralelo.
3. NumPy/OpenCV genera matrices `512x512` y RAW para CUDA.
4. Se crea el ZIP para Colab.

Salidas principales:

```text
src/salidas/tablaRutas.csv
src/salidas/npyCpu/
src/salidas/rawCuda/
src/salidas/paquetes/paqueteCudaColab.zip
src/metricas/metricasDask.csv
src/metricas/metricasCpu.csv
src/metricas/metricasPrepararCuda.csv
```

## 4. CUDA en Colab

Sube:

```text
src/salidas/paquetes/paqueteCudaColab.zip
```

Abre:

```text
src/proyecto/fases/cuda/Barahona,Chasipanta,Cañas_Cuda.ipynb
```

Ejecuta el notebook completo. El kernel aplica un filtro de enfoque con memoria compartida. Este filtro aumenta el peso del pixel central y resta parte de los vecinos inmediatos para resaltar detalles:

```text
[  0  -1   0 ]
[ -1   5  -1 ]
[  0  -1   0 ]
```

Descarga:

```text
resultadosCuda.zip
```

Colocalo en:

```text
src/salidas/paquetes/resultadosCuda.zip
```

## 5. Extraer resultados CUDA

```bash
python main.py --etapa extraer-cuda --zip-resultados src/salidas/paquetes/resultadosCuda.zip
python main.py --etapa resumen
```

Esto solo extrae y deja listos los resultados:

```text
src/salidas/cudaBin/
src/metricas/metricasCudaColab.csv
src/metricas/metricasResumen.csv
```

No hay conversion a PNG. La interfaz lee directamente los `.bin` de CUDA.

## 6. Abrir Streamlit

```bash
streamlit run appStreamlit.py
```

El dashboard muestra:

- Resumen de tiempos Dask, CPU y CUDA.
- DASK: volumen, clases y tabla de rutas.
- CPU: procesos, chunks, dimensiones y metricas.
- CUDA: tiempos de kernel, filtro de enfoque y binarios generados.
- Imagenes: original, RAW antes de CUDA y resultado CUDA.

## 7. Comandos individuales

```bash
python main.py --etapa descarga
python main.py --etapa dask
python main.py --etapa cpu
python main.py --etapa zip-cuda
python main.py --etapa extraer-cuda --zip-resultados src/salidas/paquetes/resultadosCuda.zip
python main.py --etapa resumen
```
