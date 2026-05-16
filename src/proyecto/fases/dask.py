import time
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
from dask import delayed


from proyecto.configuracion.configuracionProyecto import (
    extensionesPermitidas,
    limiteImagenesPrueba,
    rutaDatosLocal,
    rutaMetricas,
    rutaMetricasDask,
    rutaMetricasDaskClases,
    rutaSalidas,
    rutaTablaRutas,
)


def dividirLista(datos, tamanoChunk):
    """Divide una lista grande para que Dask construya particiones lazy."""
    for indice in range(0, len(datos), tamanoChunk):
        yield datos[indice: indice + tamanoChunk]


def crearMetadatosParticion(rutasTexto):
    """Convierte una particion de rutas en una tabla de metadatos."""
    registros = []
    for rutaTexto in rutasTexto:
        rutaImagen = Path(rutaTexto)
        try:
            registros.append({
                "rutaImagen": str(rutaImagen),
                "nombreArchivo": rutaImagen.name,
                "clase": rutaImagen.parent.name,
                "extension": rutaImagen.suffix.lower(),
                "tamanoBytes": int(rutaImagen.stat().st_size),
                "valida": True,
                "error": "",
            })
        except Exception as error:
            registros.append({
                "rutaImagen": str(rutaImagen),
                "nombreArchivo": rutaImagen.name,
                "clase": "",
                "extension": rutaImagen.suffix.lower(),
                "tamanoBytes": 0,
                "valida": False,
                "error": str(error),
            })
    return pd.DataFrame(registros)


def buscarRutasImagenes():
    """Lista las imagenes candidatas desde el dataset local."""
    rutasImagenes = []
    for extension in extensionesPermitidas:
        rutasImagenes.extend(rutaDatosLocal.rglob(f"*{extension}"))
        rutasImagenes.extend(rutaDatosLocal.rglob(f"*{extension.upper()}"))

    rutasImagenes = sorted(set(rutasImagenes))
    if limiteImagenesPrueba > 0:
        rutasImagenes = rutasImagenes[:limiteImagenesPrueba]
    return rutasImagenes


def ejecutarFaseDask():
    """Carga rutas con Dask DataFrame, limpia y agrega sin usar Pandas directo."""
    inicio = time.perf_counter()
    rutaSalidas.mkdir(parents=True, exist_ok=True)
    rutaMetricas.mkdir(parents=True, exist_ok=True)

    if not rutaDatosLocal.exists():
        raise FileNotFoundError(
            f"No existe el dataset en {rutaDatosLocal}. Ejecuta: python main.py --etapa descarga"
        )

    rutasImagenes = buscarRutasImagenes()
    if not rutasImagenes:
        raise ValueError("No se encontraron imagenes en el dataset.")

    meta = pd.DataFrame({
        "rutaImagen": pd.Series(dtype="string"),
        "nombreArchivo": pd.Series(dtype="string"),
        "clase": pd.Series(dtype="string"),
        "extension": pd.Series(dtype="string"),
        "tamanoBytes": pd.Series(dtype="int64"),
        "valida": pd.Series(dtype="bool"),
        "error": pd.Series(dtype="string"),
    })

    particiones = [
        delayed(crearMetadatosParticion)([str(ruta) for ruta in chunk])
        for chunk in dividirLista(rutasImagenes, 1000)
    ]

    tablaDask = dd.from_delayed(particiones, meta=meta)
    tablaDask = tablaDask.astype({
        "rutaImagen": "string",
        "nombreArchivo": "string",
        "clase": "string",
        "extension": "string",
        "tamanoBytes": "int64",
        "valida": "bool",
        "error": "string",
    })

    # Limpieza lazy: se filtran archivos invalidos y extensiones no permitidas.
    tablaFiltrada = tablaDask[
        (tablaDask["valida"])
        & (tablaDask["tamanoBytes"] > 0)
        & (tablaDask["extension"].isin(extensionesPermitidas))
    ]

    resumenClasesDask = tablaFiltrada.groupby("clase").agg({
        "rutaImagen": "count",
        "tamanoBytes": ["sum", "mean", "min", "max"],
    })

    tablaRutas, resumenClases = dd.compute(tablaFiltrada, resumenClasesDask)
    if tablaRutas.empty:
        raise ValueError("No quedaron imagenes validas despues del filtrado.")

    tablaRutas = tablaRutas.sort_values(["clase", "rutaImagen"]).reset_index(drop=True)
    tablaRutas["idImagen"] = range(len(tablaRutas))
    tablaRutas.to_csv(rutaTablaRutas, index=False)

    resumenClases.columns = [
        "cantidadImagenes",
        "tamanoTotalBytes",
        "tamanoPromedioBytes",
        "tamanoMinimoBytes",
        "tamanoMaximoBytes",
    ]
    resumenClases = resumenClases.reset_index()
    resumenClases["tamanoTotalGb"] = resumenClases["tamanoTotalBytes"] / (1024 ** 3)
    resumenClases.to_csv(rutaMetricasDaskClases, index=False)

    tiempo = time.perf_counter() - inicio
    metricasDask = pd.DataFrame([{
        "fase": "daskDataFrame",
        "descripcion": "Ingesta lazy con Dask DataFrame, limpieza, tipado y groupby por clase",
        "cantidadImagenes": int(len(tablaRutas)),
        "cantidadClases": int(tablaRutas["clase"].nunique()),
        "tamanoTotalGb": float(tablaRutas["tamanoBytes"].sum() / (1024 ** 3)),
        "tiempoSegundos": float(tiempo),
        "archivoSalida": str(rutaTablaRutas),
    }])
    metricasDask.to_csv(rutaMetricasDask, index=False)

    print("Fase Dask DataFrame completada.")
    print(metricasDask)
    print("\nResumen por clase:")
    print(resumenClases)
    return tablaRutas


if __name__ == "__main__":
    ejecutarFaseDask()
