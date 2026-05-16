# Pasos completos

## 1. Preparar entorno

```bash
cd '/home/damian/Barahona,Chasipanta,Cañas_TallerFinalU2'
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Ejecutar todo el pipeline local

```bash
python main.py
```

Esto descarga el dataset en `data/`, ejecuta Dask, procesa CPU con OpenCV, prepara RAW CUDA y genera:

```text
salidas/paquetes/paqueteCudaColab.zip
salidas/tablaRutas.csv
salidas/npyCpu/
salidas/rawCuda/
metricas/metricasDask.csv
metricas/metricasCpu.csv
metricas/metricasCpuClases.csv
metricas/metricasPrepararCuda.csv
metricas/metricasResumen.csv
```

Si no quieres borrar resultados anteriores:

```bash
python main.py --no-borrar
```

## 3. Ejecutar CUDA en Colab

Sube `salidas/paquetes/paqueteCudaColab.zip` a Colab y abre:

```text
src/proyecto/fases/cuda/Barahona,Chasipanta,Cañas_Cuda.ipynb
```

Ejecuta las celdas. El notebook:

- instala/carga `nvcc4jupyter`;
- descomprime el paquete;
- ejecuta CUDA con filtro laplaciano y memoria compartida;
- genera `metricas/metricasCudaColab.csv`;
- crea y descarga `resultadosCuda.zip`.

## 4. Integrar resultados CUDA

Guarda `resultadosCuda.zip` como `salidas/paquetes/resultadosCuda.zip` y ejecuta:

```bash
python main.py --etapa integrar-cuda --zip-resultados salidas/paquetes/resultadosCuda.zip
```

Esto genera:

```text
salidas/cudaNpy/
salidas/cudaPng/
metricas/metricasIntegracionCuda.csv
metricas/metricasResumen.csv
```

## 5. Abrir dashboard

```bash
streamlit run appStreamlit.py
```

## 6. Etapas individuales

```bash
python main.py --etapa descarga
python main.py --etapa dask
python main.py --etapa cpu --procesos 8
python main.py --etapa zip-cuda
python main.py --etapa integrar-cuda --zip-resultados salidas/paquetes/resultadosCuda.zip
python main.py --etapa resumen
```
