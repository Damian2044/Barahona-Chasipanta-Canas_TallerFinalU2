import shutil
from pathlib import Path

import kagglehub

from proyecto.configuracion.configuracionProyecto import rutaData, rutaDatasetProyecto, slugDatasetKaggle


def descargarDatasetSiHaceFalta():
    """Descarga el dataset en data/ si aun no existe en el proyecto."""
    if rutaDatasetProyecto.exists() and any(rutaDatasetProyecto.rglob("*")):
        print(f"Dataset existente: {rutaDatasetProyecto}")
        return rutaDatasetProyecto

    rutaData.mkdir(parents=True, exist_ok=True)
    rutaCache = Path(kagglehub.dataset_download(slugDatasetKaggle))

    if rutaDatasetProyecto.exists():
        shutil.rmtree(rutaDatasetProyecto)

    shutil.copytree(rutaCache, rutaDatasetProyecto)
    print(f"Dataset copiado a: {rutaDatasetProyecto}")
    return rutaDatasetProyecto


if __name__ == "__main__":
    descargarDatasetSiHaceFalta()
