import shutil
import zipfile
from pathlib import Path

from proyecto.configuracion.configuracionProyecto import (
    rutaBaseProyecto,
    rutaCudaBin,
    rutaCudaNpy,
    rutaCudaPng,
    rutaMetricas,
    rutaNpyCpu,
    rutaPaquetes,
    rutaRawCuda,
    rutaSalidas,
    rutaZipCudaEntrada,
    rutaZipCudaResultados,
)


def crearCarpetasProyecto():
    """Crea las carpetas base que usa el pipeline."""
    for ruta in [rutaSalidas, rutaMetricas, rutaPaquetes, rutaNpyCpu, rutaRawCuda, rutaCudaBin, rutaCudaNpy, rutaCudaPng]:
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

    return rutaZipCudaEntrada


def descomprimirZipResultados(rutaZip):
    """Extrae el zip de resultados CUDA dentro del proyecto."""
    rutaZip = Path(rutaZip)
    if not rutaZip.exists():
        raise FileNotFoundError(f"No existe el zip de resultados CUDA: {rutaZip}")
    with zipfile.ZipFile(rutaZip, "r") as zipEntrada:
        zipEntrada.extractall(rutaBaseProyecto)
