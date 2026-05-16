import os
from pathlib import Path

###### RUTAS BASE DEL PROYECTO ######
rutaBaseProyecto = Path(__file__).resolve().parents[2]
rutaData = rutaBaseProyecto / "data"
rutaSalidas = rutaBaseProyecto / "salidas"
rutaMetricas = rutaBaseProyecto / "metricas"

###### DATASET DE KAGGLE ######
nombreDataset = "Corn Leaf Disease"
enlaceDatasetKaggle = "https://www.kaggle.com/datasets/ndisan/corn-leaf-disease"
slugDatasetKaggle = "ndisan/corn-leaf-disease"
nombreCarpetaDataset = "corn-leaf-disease"
rutaDatasetProyecto = rutaData / nombreCarpetaDataset

# DATASET_RUTA permite usar una ubicacion externa si el docente ya lo tiene descargado.
rutaDatasetEntorno = os.environ.get("DATASET_RUTA", "").strip()
rutaDatosLocal = Path(rutaDatasetEntorno) if rutaDatasetEntorno else rutaDatasetProyecto

###### SALIDAS DEL PIPELINE ######
rutaTablaRutas = rutaSalidas / "tablaRutas.csv"
rutaNpyCpu = rutaSalidas / "npyCpu"
rutaRawCuda = rutaSalidas / "rawCuda"
rutaCudaBin = rutaSalidas / "cudaBin"
rutaCudaNpy = rutaSalidas / "cudaNpy"
rutaCudaPng = rutaSalidas / "cudaPng"
rutaPaquetes = rutaSalidas / "paquetes"

###### ZIPS PARA COLAB ######
rutaZipCudaEntrada = rutaPaquetes / "paqueteCudaColab.zip"
rutaZipCudaResultados = rutaPaquetes / "resultadosCuda.zip"

###### METRICAS ######
rutaMetricasDask = rutaMetricas / "metricasDask.csv"
rutaMetricasDaskClases = rutaMetricas / "metricasDaskClases.csv"
rutaMetricasCpu = rutaMetricas / "metricasCpu.csv"
rutaMetricasCpuClases = rutaMetricas / "metricasCpuClases.csv"
rutaMetricasPrepararCuda = rutaMetricas / "metricasPrepararCuda.csv"
rutaMetricasCudaColab = rutaMetricas / "metricasCudaColab.csv"
rutaMetricasIntegracionCuda = rutaMetricas / "metricasIntegracionCuda.csv"
rutaMetricasResumen = rutaMetricas / "metricasResumen.csv"

###### PARAMETROS DEL PROCESAMIENTO LOCAL ######
tamanoObjetivo = 512
limiteImagenesPrueba = 0  # 0 = usar todas las imagenes.

clasesEsperadas = [
    "Bercak Daun",
    "Daun Sehat",
    "Hawar Daun",
    "Karat Daun",
]

extensionesPermitidas = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
