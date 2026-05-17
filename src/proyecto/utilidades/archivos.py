import shutil
import time
import zipfile
from pathlib import Path

import pandas as pd

from proyecto.configuracion.configuracionProyecto import (
    rutaBaseProyecto,
    rutaCudaBin,
    rutaMetricas,
    rutaMetricasExtraccionCuda,
    rutaMetricasPaqueteCuda,
    rutaNpyCpu,
    rutaPaquetes,
    rutaRawCuda,
    rutaSalidas,
    rutaZipCudaEntrada,
    rutaZipCudaResultados,
)


def crearCarpetasProyecto():
    """Crea las carpetas base que usa el pipeline."""
    for ruta in [rutaSalidas, rutaMetricas, rutaPaquetes, rutaNpyCpu, rutaRawCuda, rutaCudaBin]:
        ruta.mkdir(parents=True, exist_ok=True)


def limpiarResultados():
    """Limpia salidas regenerables, conservando data/ y el codigo fuente."""
    for ruta in [rutaSalidas, rutaMetricas, rutaZipCudaEntrada, rutaZipCudaResultados]:
        if ruta.is_dir():
            shutil.rmtree(ruta)
        elif ruta.exists():
            ruta.unlink()
    crearCarpetasProyecto()


def crearZipCudaEntrada():
    """Empaqueta RAW y metricas para subir directamente a Colab."""
    inicio = time.perf_counter()
    if rutaZipCudaEntrada.exists():
        rutaZipCudaEntrada.unlink()

    rutasRequeridas = [
        rutaBaseProyecto / "salidas" / "rawCuda",
        rutaBaseProyecto / "metricas" / "metricasPrepararCuda.csv",
    ]

    with zipfile.ZipFile(rutaZipCudaEntrada, "w", compression=zipfile.ZIP_DEFLATED) as zipSalida:
        for ruta in rutasRequeridas:
            if ruta.is_file():
                zipSalida.write(ruta, ruta.relative_to(rutaBaseProyecto))
            elif ruta.is_dir():
                for archivo in ruta.rglob("*"):
                    if archivo.is_file():
                        zipSalida.write(archivo, archivo.relative_to(rutaBaseProyecto))

    tiempo = time.perf_counter() - inicio
    pd.DataFrame([{
        "fase": "paqueteCuda",
        "descripcion": "Creacion del ZIP con RAW y metricas para Colab",
        "rutaZip": rutaZipCudaEntrada.relative_to(rutaBaseProyecto).as_posix(),
        "tamanoZipMb": rutaZipCudaEntrada.stat().st_size / (1024 ** 2) if rutaZipCudaEntrada.exists() else 0.0,
        "tiempoSegundos": float(tiempo),
    }]).to_csv(rutaMetricasPaqueteCuda, index=False)

    return rutaZipCudaEntrada


def descomprimirZipResultados(rutaZip):
    """Extrae el zip de resultados CUDA dentro del proyecto."""
    inicio = time.perf_counter()
    rutaZip = Path(rutaZip)
    if not rutaZip.exists():
        raise FileNotFoundError(f"No existe el zip de resultados CUDA: {rutaZip}")
    with zipfile.ZipFile(rutaZip, "r") as zipEntrada:
        zipEntrada.extractall(rutaBaseProyecto)

    tiempo = time.perf_counter() - inicio
    pd.DataFrame([{
        "fase": "extraerResultadosCuda",
        "descripcion": "Extraccion del ZIP descargado desde Colab",
        "rutaZip": rutaZip.relative_to(rutaBaseProyecto).as_posix() if rutaZip.is_absolute() else rutaZip.as_posix(),
        "tiempoSegundos": float(tiempo),
    }]).to_csv(rutaMetricasExtraccionCuda, index=False)
    return rutaBaseProyecto
