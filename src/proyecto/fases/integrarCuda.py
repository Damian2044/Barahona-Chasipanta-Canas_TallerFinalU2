from pathlib import Path

import pandas as pd

from proyecto.configuracion.configuracionProyecto import (
    rutaCudaBin,
    rutaMetricas,
    rutaMetricasCudaColab,
    rutaZipCudaResultados,
)
from proyecto.utilidades.archivos import descomprimirZipResultados


def extraerResultadosCuda(rutaZip=None):
    """Descomprime el ZIP de Colab y deja los binarios listos para Streamlit."""
    rutaMetricas.mkdir(parents=True, exist_ok=True)

    rutaZip = Path(rutaZip) if rutaZip else rutaZipCudaResultados
    descomprimirZipResultados(rutaZip)

    if not rutaMetricasCudaColab.exists():
        raise FileNotFoundError(
            "Se extrajo el ZIP, pero no existe metricasCudaColab.csv."
        )
    if not rutaCudaBin.exists():
        raise FileNotFoundError("Se extrajo el ZIP, pero no existe salidas/cudaBin.")

    metricasCuda = pd.read_csv(rutaMetricasCudaColab)
    print("Resultados CUDA extraidos.")
    print(f"Metricas CUDA: {rutaMetricasCudaColab}")
    print(f"Binarios CUDA: {rutaCudaBin}")
    print(f"Imagenes reportadas por CUDA: {len(metricasCuda)}")
    return metricasCuda


if __name__ == "__main__":
    extraerResultadosCuda()
