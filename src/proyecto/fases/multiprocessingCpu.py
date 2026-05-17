import time
from math import ceil
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


from proyecto.configuracion.configuracionProyecto import (
    rutaBaseProyecto,
    rutaMetricas,
    rutaMetricasCpu,
    rutaMetricasCpuClases,
    rutaMetricasPrepararCuda,
    rutaNpyCpu,
    rutaRawCuda,
    rutaTablaRutas,
    tamanoObjetivo,
)
from proyecto.utilidades.archivos import crearZipCudaEntrada


def dividirRegistros(registros, tamanoChunk):
    """Crea chunks para repartir trabajo pesado entre procesos."""
    for indice in range(0, len(registros), tamanoChunk):
        yield registros[indice: indice + tamanoChunk]


def resolverProcesos(procesosSolicitados=None):
    """Usa todos los CPU disponibles cuando no se define un numero fijo."""
    disponibles = cpu_count()
    if procesosSolicitados is None or procesosSolicitados <= 0:
        return disponibles
    return min(procesosSolicitados, disponibles)


def calcularTamanoChunk(cantidadRegistros, procesosUsados):
    """Divide el trabajo en un chunk principal por proceso."""
    if cantidadRegistros <= 0:
        return 1
    return max(1, ceil(cantidadRegistros / max(procesosUsados, 1)))


def procesarChunkCpu(argumentos):
    """Procesa un chunk completo dentro de un proceso del Pool."""
    indiceChunk, registros = argumentos
    resultadosCpu = []
    resultadosCuda = []

    for fila in registros:
        resultadoCpu, resultadoCuda = procesarImagenCpu(fila, indiceChunk)
        resultadosCpu.append(resultadoCpu)
        resultadosCuda.append(resultadoCuda)

    return resultadosCpu, resultadosCuda


def procesarImagenCpu(fila, indiceChunk=0):
    """Lee con OpenCV, redimensiona a 512x512 y deja RAW listo para CUDA."""
    inicio = time.perf_counter()
    idImagen = int(fila["idImagen"])
    rutaImagenCsv = Path(str(fila["rutaImagen"]))
    rutaImagen = rutaImagenCsv if rutaImagenCsv.is_absolute() else rutaBaseProyecto / rutaImagenCsv
    clase = str(fila["clase"])
    tamanoOriginalBytes = int(fila.get("tamanoBytes", 0))

    rutaClaseNpy = rutaNpyCpu / clase
    rutaClaseRaw = rutaRawCuda / clase
    rutaClaseNpy.mkdir(parents=True, exist_ok=True)
    rutaClaseRaw.mkdir(parents=True, exist_ok=True)

    rutaSalidaNpy = rutaClaseNpy / f"imagen_{idImagen}.npy"
    rutaSalidaRaw = rutaClaseRaw / f"imagen_{idImagen}_raw.bin"
    rutaOriginalRelativa = rutaImagen.relative_to(rutaBaseProyecto).as_posix()
    rutaSalidaNpyRelativa = rutaSalidaNpy.relative_to(rutaBaseProyecto).as_posix()
    rutaSalidaRawRelativa = rutaSalidaRaw.relative_to(rutaBaseProyecto).as_posix()

    try:
        imagenColor = cv2.imread(str(rutaImagen), cv2.IMREAD_COLOR)
        if imagenColor is None:
            raise ValueError("OpenCV no pudo leer la imagen")

        altoOriginal, anchoOriginal = imagenColor.shape[:2]
        canalesOriginales = int(imagenColor.shape[2]) if len(imagenColor.shape) == 3 else 1

        imagenGris = cv2.cvtColor(imagenColor, cv2.COLOR_BGR2GRAY)
        imagenResize = cv2.resize(
            imagenGris,
            (tamanoObjetivo, tamanoObjetivo),
            interpolation=cv2.INTER_AREA,
        )

        matrizNormalizada = imagenResize.astype(np.float32) / 255.0
        np.save(rutaSalidaNpy, matrizNormalizada)
        imagenResize.astype(np.uint8).tofile(rutaSalidaRaw)

        tiempo = time.perf_counter() - inicio
        tamanoResizeBytes = int(imagenResize.nbytes)
        pixelesOriginales = int(altoOriginal * anchoOriginal)
        pixelesResize = int(tamanoObjetivo * tamanoObjetivo)

        resultadoCpu = {
            "idImagen": idImagen,
            "clase": clase,
            "rutaOriginal": rutaOriginalRelativa,
            "rutaNpy": rutaSalidaNpyRelativa,
            "rutaRaw": rutaSalidaRawRelativa,
            "altoOriginal": int(altoOriginal),
            "anchoOriginal": int(anchoOriginal),
            "canalesOriginales": canalesOriginales,
            "pixelesOriginales": pixelesOriginales,
            "tamanoOriginalBytes": tamanoOriginalBytes,
            "altoResize": tamanoObjetivo,
            "anchoResize": tamanoObjetivo,
            "pixelesResize": pixelesResize,
            "tamanoResizeBytes": tamanoResizeBytes,
            "relacionPixelesResize": float(pixelesResize / pixelesOriginales) if pixelesOriginales else 0.0,
            "media": float(np.mean(matrizNormalizada)),
            "desviacion": float(np.std(matrizNormalizada)),
            "minimo": float(np.min(matrizNormalizada)),
            "maximo": float(np.max(matrizNormalizada)),
            "mediana": float(np.median(matrizNormalizada)),
            "percentil25": float(np.percentile(matrizNormalizada, 25)),
            "percentil75": float(np.percentile(matrizNormalizada, 75)),
            "tiempoSegundos": float(tiempo),
            "indiceChunk": int(indiceChunk),
            "estado": "ok",
            "error": "",
        }

        resultadoCuda = {
            "idImagen": idImagen,
            "clase": clase,
            "rutaRaw": rutaSalidaRawRelativa,
            "alto": tamanoObjetivo,
            "ancho": tamanoObjetivo,
            "tipoDato": "uint8",
            "cantidadElementos": pixelesResize,
            "tamanoBytes": tamanoResizeBytes,
            "estado": "ok",
            "error": "",
        }

    except Exception as error:
        tiempo = time.perf_counter() - inicio
        resultadoCpu = {
            "idImagen": idImagen,
            "clase": clase,
            "rutaOriginal": rutaOriginalRelativa,
            "rutaNpy": "",
            "rutaRaw": "",
            "altoOriginal": 0,
            "anchoOriginal": 0,
            "canalesOriginales": 0,
            "pixelesOriginales": 0,
            "tamanoOriginalBytes": tamanoOriginalBytes,
            "altoResize": tamanoObjetivo,
            "anchoResize": tamanoObjetivo,
            "pixelesResize": tamanoObjetivo * tamanoObjetivo,
            "tamanoResizeBytes": 0,
            "relacionPixelesResize": 0.0,
            "media": 0.0,
            "desviacion": 0.0,
            "minimo": 0.0,
            "maximo": 0.0,
            "mediana": 0.0,
            "percentil25": 0.0,
            "percentil75": 0.0,
            "tiempoSegundos": float(tiempo),
            "indiceChunk": int(indiceChunk),
            "estado": "error",
            "error": str(error),
        }
        resultadoCuda = {
            "idImagen": idImagen,
            "clase": clase,
            "rutaRaw": "",
            "alto": 0,
            "ancho": 0,
            "tipoDato": "uint8",
            "cantidadElementos": 0,
            "tamanoBytes": 0,
            "estado": "error",
            "error": str(error),
        }

    return resultadoCpu, resultadoCuda


def crearMetricasCpuPorClase(metricasCpu):
    """Resume dimensiones, tamanos y tiempos por clase para Streamlit."""
    datosOk = metricasCpu[metricasCpu["estado"] == "ok"].copy()
    if datosOk.empty:
        return pd.DataFrame()

    resumen = (
        datosOk
        .groupby("clase")
        .agg(
            cantidadImagenes=("idImagen", "count"),
            altoOriginalPromedio=("altoOriginal", "mean"),
            anchoOriginalPromedio=("anchoOriginal", "mean"),
            tamanoOriginalMbPromedio=("tamanoOriginalBytes", lambda serie: serie.mean() / (1024 ** 2)),
            tamanoResizeKbPromedio=("tamanoResizeBytes", lambda serie: serie.mean() / 1024),
            mediaPromedio=("media", "mean"),
            desviacionPromedio=("desviacion", "mean"),
            tiempoPromedioSegundos=("tiempoSegundos", "mean"),
        )
        .reset_index()
    )
    return resumen


def ejecutarFaseMultiprocessing(procesos=None, crearZip=True):
    """Procesa imagenes con Pool, chunks y OpenCV; tambien prepara RAW CUDA."""
    inicioGlobal = time.perf_counter()
    rutaNpyCpu.mkdir(parents=True, exist_ok=True)
    rutaRawCuda.mkdir(parents=True, exist_ok=True)
    rutaMetricas.mkdir(parents=True, exist_ok=True)

    if not rutaTablaRutas.exists():
        raise FileNotFoundError("No existe tablaRutas.csv. Ejecuta primero la fase Dask.")

    tablaRutas = pd.read_csv(rutaTablaRutas)
    registros = tablaRutas.to_dict(orient="records")
    procesosUsados = resolverProcesos(procesos)
    tamanoChunk = calcularTamanoChunk(len(registros), procesosUsados)
    chunks = list(enumerate(dividirRegistros(registros, tamanoChunk)))

    resultadosCpu = []
    resultadosCuda = []

    with Pool(processes=procesosUsados) as pool:
        for cpuChunk, cudaChunk in tqdm(
            pool.imap_unordered(procesarChunkCpu, chunks),
            total=len(chunks),
            desc="Multiprocessing CPU por chunks",
        ):
            resultadosCpu.extend(cpuChunk)
            resultadosCuda.extend(cudaChunk)

    tiempoGlobal = time.perf_counter() - inicioGlobal

    metricasCpu = pd.DataFrame(resultadosCpu).sort_values("idImagen")
    metricasCpu["procesosUsados"] = procesosUsados
    metricasCpu["cantidadChunks"] = len(chunks)
    metricasCpu["tamanoChunk"] = tamanoChunk
    metricasCpu["tiempoGlobalSegundos"] = float(tiempoGlobal)
    metricasCpu.to_csv(rutaMetricasCpu, index=False)

    metricasPrepararCuda = pd.DataFrame(resultadosCuda).sort_values("idImagen")
    metricasPrepararCuda["tiempoPreparacionGlobalSegundos"] = float(tiempoGlobal)
    metricasPrepararCuda.to_csv(rutaMetricasPrepararCuda, index=False)

    metricasCpuClases = crearMetricasCpuPorClase(metricasCpu)
    metricasCpuClases.to_csv(rutaMetricasCpuClases, index=False)

    rutaZip = crearZipCudaEntrada() if crearZip else None

    print("Fase Multiprocessing completada.")
    print(f"Procesos usados: {procesosUsados}")
    print(f"Chunks procesados: {len(chunks)}")
    print(f"Tiempo total real: {tiempoGlobal:.2f} segundos")
    print(f"Metricas CPU: {rutaMetricasCpu}")
    print(f"Metricas CUDA entrada: {rutaMetricasPrepararCuda}")
    if rutaZip:
        print(f"ZIP listo para Colab: {rutaZip}")
    return metricasCpu


if __name__ == "__main__":
    ejecutarFaseMultiprocessing()
