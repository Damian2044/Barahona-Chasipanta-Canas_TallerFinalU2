import pandas as pd

from proyecto.configuracion.configuracionProyecto import (
    rutaMetricas,
    rutaMetricasCpu,
    rutaMetricasCudaColab,
    rutaMetricasDask,
    rutaMetricasResumen,
)


def tiempoUnicoSiExiste(tabla, columna, alternativa):
    """Evita tiempos raros: usa tiempo real de fase cuando esta disponible."""
    if columna in tabla.columns and not tabla.empty:
        return float(tabla[columna].dropna().iloc[0])
    return float(alternativa)


def crearMetricasResumen():
    rutaMetricas.mkdir(parents=True, exist_ok=True)
    registros = []

    if rutaMetricasDask.exists():
        dask = pd.read_csv(rutaMetricasDask)
        registros.append({
            "fase": "Dask DataFrame",
            "tiempoSegundos": float(dask["tiempoSegundos"].sum()),
            "cantidadRegistros": int(dask["cantidadImagenes"].sum()),
            "detalle": "Ingesta lazy, limpieza y groupby",
        })

    if rutaMetricasCpu.exists():
        cpu = pd.read_csv(rutaMetricasCpu)
        registros.append({
            "fase": "Multiprocessing CPU + RAW",
            "tiempoSegundos": tiempoUnicoSiExiste(cpu, "tiempoGlobalSegundos", cpu["tiempoSegundos"].sum()),
            "cantidadRegistros": int(len(cpu)),
            "detalle": "OpenCV, resize 512x512, metricas y RAW CUDA",
        })

    if rutaMetricasCudaColab.exists():
        cuda = pd.read_csv(rutaMetricasCudaColab)
        if "tiempoTotalGpuSegundos" in cuda.columns:
            tiempoCuda = float(cuda["tiempoTotalGpuSegundos"].sum())
        elif "tiempoKernelMs" in cuda.columns:
            tiempoCuda = float(cuda["tiempoKernelMs"].sum()) / 1000.0
        else:
            tiempoCuda = 0.0
        registros.append({
            "fase": "CUDA GPU Enfoque",
            "tiempoSegundos": tiempoCuda,
            "cantidadRegistros": int(len(cuda)),
            "detalle": "Filtro de enfoque 3x3 en GPU con memoria compartida",
        })

    resumen = pd.DataFrame(registros)
    resumen.to_csv(rutaMetricasResumen, index=False)
    print(resumen.to_string(index=False))
    print(f"Metricas resumen guardadas en: {rutaMetricasResumen}")
    return resumen


if __name__ == "__main__":
    crearMetricasResumen()
