# Procesamiento a gran escala con arquitectura hibrida

Pipeline del taller:

```text
Kaggle -> Dask -> Multiprocessing CPU + NumPy/RAW -> CUDA en Colab -> Extraccion ZIP -> Streamlit
```

Dataset:

<https://www.kaggle.com/datasets/ndisan/corn-leaf-disease>

El dataset contiene 4000 imagenes RGB de hojas de maiz, distribuidas en 4 clases balanceadas de 1000 imagenes cada una.

## 1. Preparar entorno

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Descargar dataset

```bash
python main.py --etapa descarga
```

El dataset queda en:

```text
src/data/corn-leaf-disease/
```

## 3. Ejecutar procesamiento local

```bash
python main.py --no-borrar
```

Esto ejecuta:

- Dask: genera `src/salidas/tablaRutas.csv` y metricas por clase.
- Multiprocessing CPU: lee imagenes, convierte a gris, redimensiona a `512x512`, calcula metricas y genera RAW para CUDA.
- ZIP CUDA: crea `src/salidas/paquetes/paqueteCudaColab.zip` para ejecutar el filtro de enfoque en GPU.

## 4. Ejecutar CUDA en Colab

Sube a Colab:

```text
src/salidas/paquetes/paqueteCudaColab.zip
```

Abre y ejecuta:

```text
src/proyecto/fases/cuda/Barahona,Chasipanta,Cañas_Cuda.ipynb
```

El kernel CUDA aplica un filtro de enfoque con memoria compartida. Este filtro realza bordes y detalles porque aumenta el peso del pixel central y resta parte de sus vecinos inmediatos:

```text
[  0  -1   0 ]
[ -1   5  -1 ]
[  0  -1   0 ]
```

Descarga el archivo generado por Colab:

```text
resultadosCuda.zip
```

Guarda ese ZIP en:

```text
src/salidas/paquetes/resultadosCuda.zip
```

## 5. Extraer resultados CUDA

```bash
python main.py --etapa extraer-cuda --zip-resultados src/salidas/paquetes/resultadosCuda.zip
python main.py --etapa resumen
```

La extraccion deja disponibles:

```text
src/salidas/cudaBin/
src/metricas/metricasCudaColab.csv
src/metricas/metricasExtraccionCuda.csv
src/metricas/metricasResumen.csv
```

Streamlit lee directamente los binarios `.bin` generados por CUDA. No se requiere convertir a PNG.

## 6. Abrir dashboard

```bash
streamlit run appStreamlit.py
```

## Comandos por etapa

```bash
python main.py --etapa descarga
python main.py --etapa dask
python main.py --etapa cpu
python main.py --etapa zip-cuda
python main.py --etapa extraer-cuda --zip-resultados src/salidas/paquetes/resultadosCuda.zip
python main.py --etapa resumen
```

## Inicio rapido desde el repositorio

Clona el repositorio y entra a la carpeta del proyecto:

```bash
git clone https://github.com/Damian2044/Barahona-Chasipanta-Canas_TallerFinalU2.git
cd Barahona-Chasipanta-Canas_TallerFinalU2
```

Crea el entorno virtual e instala las dependencias:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

El repositorio debe subir los CSV de `src/metricas/` y los ZIP de `src/salidas/paquetes/`. No se suben las imagenes ni carpetas pesadas como `src/data/`, `src/salidas/rawCuda/`, `src/salidas/npyCpu/` o `src/salidas/cudaBin/`.

Descarga y extrae el dataset en tu maquina para recuperar las imagenes originales:

```bash
python main.py --etapa descarga
```

El dataset queda extraido en:

```text
src/data/corn-leaf-disease/
```

Si el ZIP de resultados CUDA ya viene en el repositorio, debe estar en:

```text
src/salidas/paquetes/resultadosCuda.zip
```

Extrae los binarios desde ese ZIP:

```bash
python main.py --etapa extraer-cuda --zip-resultados src/salidas/paquetes/resultadosCuda.zip
```

Prende el dashboard:

```bash
streamlit run appStreamlit.py
```
