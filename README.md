# Procesamiento a gran escala con arquitectura híbrida

## 1. Información del dataset

Dataset: Corn Leaf Disease

```text
https://www.kaggle.com/datasets/ndisan/corn-leaf-disease
```

El dataset contiene 4000 imágenes RGB de hojas de maíz, distribuidas en 4 clases balanceadas de 1000 imágenes cada una.

Clases:

- `Bercak Daun`: hojas con manchas foliares, asociadas a zonas grises o lesiones visibles.
- `Daun Sehat`: hojas sanas, sin síntomas claros de enfermedad.
- `Hawar Daun`: hojas con tizón o daño extendido, normalmente con zonas secas o quemadas.
- `Karat Daun`: hojas con roya, caracterizada por manchas de tono óxido o rojizo.

Distribución:

```text
Bercak Daun  -> 1000 imágenes
Daun Sehat   -> 1000 imágenes
Hawar Daun   -> 1000 imágenes
Karat Daun   -> 1000 imágenes
Total        -> 4000 imágenes
```

Pipeline:

```text
Kaggle -> Dask -> Multiprocessing CPU -> RAW/NumPy -> CUDA Colab -> Streamlit
```

## 2. Estructura del proyecto

```text
Barahona-Chasipanta-Canas_TallerFinalU2/
|-- appStreamlit.py
|-- main.py
|-- README.md
|-- requirements.txt
|-- src/
|   |-- data/                         # Dataset descargado localmente
|   |-- metricas/                     # CSV de métricas
|   |-- salidas/
|   |   |-- tablaRutas.csv
|   |   |-- npyCpu/                   # Matrices NumPy generadas por CPU
|   |   |-- rawCuda/                  # RAW 512x512 enviados a CUDA
|   |   |-- cudaBin/                  # Binarios generados por CUDA
|   |   |-- paquetes/
|   |       |-- paqueteCudaColab.zip
|   |       |-- resultadosCuda.zip
|   |-- proyecto/
|       |-- configuracion/
|       |-- fases/
|       |-- metricas/
|       |-- utilidades/
```

## 3. Archivos necesarios para Streamlit

Para ver solo métricas y gráficas:

```text
src/metricas/
src/salidas/tablaRutas.csv
```

Para ver la comparación visual completa:

```text
src/data/corn-leaf-disease/   # Imagen original
src/salidas/rawCuda/          # RAW antes de CUDA
src/salidas/cudaBin/          # Resultado CUDA enfoque
```

Streamlit no guarda imágenes dentro de los CSV. Los CSV guardan rutas relativas y la interfaz abre los archivos físicos desde esas carpetas.

## 4. Clonación y ejecución completa

Instala los requerimientos:

```bash
git clone https://github.com/Damian2044/Barahona-Chasipanta-Canas_TallerFinalU2.git
cd Barahona-Chasipanta-Canas_TallerFinalU2
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Puedes ejecutar el procesamiento local completo con:

```bash
python main.py
```

Ese comando descarga el dataset si falta, ejecuta Dask, procesa CPU, genera RAW/NPY, crea el paquete para Colab y genera el resumen disponible hasta ese punto.

También puedes ejecutar el mismo flujo por etapas:

```bash
python main.py --etapa descarga
python main.py --etapa dask
python main.py --etapa cpu
```

La etapa CPU crea:

```text
src/salidas/rawCuda/
src/salidas/npyCpu/
src/salidas/paquetes/paqueteCudaColab.zip
```

Comandos por etapa disponibles:

```bash
python main.py --etapa descarga
python main.py --etapa dask
python main.py --etapa cpu
python main.py --etapa zip-cuda
python main.py --etapa extraer-cuda --zip-resultados src/salidas/paquetes/resultadosCuda.zip
python main.py --etapa resumen
```

Sube a Colab:

```text
src/salidas/paquetes/paqueteCudaColab.zip
```

Abre el notebook:

```text
src/proyecto/fases/cuda/Barahona,Chasipanta,Cañas_Cuda.ipynb
```

Ejecuta el notebook completo. El kernel aplica el filtro de enfoque:

```text
[  0  -1   0 ]
[ -1   5  -1 ]
[  0  -1   0 ]
```

Descarga desde Colab:

```text
resultadosCuda.zip
```

Colócalo en:

```text
src/salidas/paquetes/resultadosCuda.zip
```

Extrae resultados CUDA:

```bash
python main.py --etapa extraer-cuda --zip-resultados src/salidas/paquetes/resultadosCuda.zip
```

Genera resumen final:

```bash
python main.py --etapa resumen
```

El resumen final crea `src/metricas/metricasResumen.csv`, que junta los tiempos principales de Dask, CPU y CUDA para mostrarlos en el dashboard.

Abre Streamlit:

```bash
streamlit run appStreamlit.py
```

## 5. Replicar solo estadísticas

Este modo usa los CSV que vienen en el repositorio. Sirve para revisar tablas, gráficas y tiempos sin descargar imágenes pesadas.

```bash
git clone https://github.com/Damian2044/Barahona-Chasipanta-Canas_TallerFinalU2.git
cd Barahona-Chasipanta-Canas_TallerFinalU2
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run appStreamlit.py
```

La pestaña de imágenes puede mostrar archivos como no disponibles si no existen `src/data/`, `src/salidas/rawCuda/` o `src/salidas/cudaBin/`.

## 6. Replicar con imágenes y Drive

Descarga el dataset para recuperar las imágenes originales:

```bash
python main.py --etapa descarga
```

Descarga los resultados CUDA desde Drive:

```text
https://drive.google.com/file/d/10M0TAgktOhAy-pHdDCdmss3TbZ_PddLy/view?usp=sharing
```

Coloca la carpeta descargada en:

```text
src/salidas/cudaBin/
```

Si también descargaste `rawCuda`, colócalo en:

```text
src/salidas/rawCuda/
```

Luego abre Streamlit:

```bash
streamlit run appStreamlit.py
```
