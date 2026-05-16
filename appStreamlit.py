from pathlib import Path
import sys

RUTA_SRC = Path(__file__).resolve().parent / "src"
if str(RUTA_SRC) not in sys.path:
    sys.path.insert(0, str(RUTA_SRC))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from proyecto.configuracion.configuracionProyecto import (
    enlaceDatasetKaggle,
    nombreDataset,
    rutaCudaNpy,
    rutaCudaPng,
    rutaDatosLocal,
    rutaMetricasCpu,
    rutaMetricasCpuClases,
    rutaMetricasCudaColab,
    rutaMetricasDask,
    rutaMetricasDaskClases,
    rutaMetricasIntegracionCuda,
    rutaMetricasPrepararCuda,
    rutaMetricasResumen,
    rutaTablaRutas,
    rutaZipCudaEntrada,
    tamanoObjetivo,
)

st.set_page_config(page_title="Pipeline Hibrido Imagenes", layout="wide")
st.title("Pipeline Hibrido de Imagenes")


def cargarCsvSeguro(ruta):
    ruta = Path(ruta)
    if ruta.exists():
        return pd.read_csv(ruta)
    return pd.DataFrame()


def mostrarMetrica(columna, titulo, valor, sufijo=""):
    if pd.isna(valor):
        valor = 0
    columna.metric(titulo, f"{valor}{sufijo}")


def obtenerRutaOriginal(idImagen):
    """Busca la ruta original en las metricas CPU o en tablaRutas."""
    for tabla, columna in [(metricasCpu, "rutaOriginal"), (tablaRutas, "rutaImagen")]:
        if not tabla.empty and columna in tabla.columns:
            fila = tabla[tabla["idImagen"] == idImagen]
            if not fila.empty:
                ruta = Path(str(fila[columna].iloc[0]))
                if ruta.exists():
                    return ruta
    return None


def obtenerSalidaLaplaciano(idImagen, clase):
    """Usa PNG CUDA si existe; si no, intenta el NPY CUDA."""
    rutaPng = rutaCudaPng / clase / f"imagen_{idImagen}_cuda.png"
    if rutaPng.exists():
        return "png", rutaPng

    rutaNpy = rutaCudaNpy / clase / f"imagen_{idImagen}_cuda.npy"
    if rutaNpy.exists():
        return "npy", rutaNpy

    if not metricasIntegracionCuda.empty and "rutaCudaPng" in metricasIntegracionCuda.columns:
        fila = metricasIntegracionCuda[metricasIntegracionCuda["idImagen"] == idImagen]
        if not fila.empty:
            ruta = Path(str(fila["rutaCudaPng"].iloc[0]))
            if ruta.exists():
                return "png", ruta
    return None, None


def cargarImagenNpy(rutaNpy):
    """Convierte NPY de CUDA a imagen uint8 para Streamlit."""
    matriz = np.asarray(np.load(rutaNpy), dtype=np.float32)
    if matriz.size and matriz.max() <= 1.0:
        matriz = matriz * 255.0
    return np.clip(matriz, 0, 255).astype(np.uint8)


metricasDask = cargarCsvSeguro(rutaMetricasDask)
metricasDaskClases = cargarCsvSeguro(rutaMetricasDaskClases)
metricasCpu = cargarCsvSeguro(rutaMetricasCpu)
metricasCpuClases = cargarCsvSeguro(rutaMetricasCpuClases)
metricasPrepararCuda = cargarCsvSeguro(rutaMetricasPrepararCuda)
metricasCudaColab = cargarCsvSeguro(rutaMetricasCudaColab)
metricasIntegracionCuda = cargarCsvSeguro(rutaMetricasIntegracionCuda)
metricasResumen = cargarCsvSeguro(rutaMetricasResumen)
tablaRutas = cargarCsvSeguro(rutaTablaRutas)

st.caption(
    f"Dataset: {nombreDataset} | Kaggle: {enlaceDatasetKaggle} | "
    f"Ruta local: {rutaDatosLocal} | Resize: {tamanoObjetivo}x{tamanoObjetivo}"
)

tabResumen, tabDataset, tabCpu, tabCuda, tabImagenes = st.tabs([
    "Resumen",
    "Dataset",
    "CPU y RAW",
    "CUDA",
    "Imagenes",
])

with tabResumen:
    st.subheader("Estado general")
    col1, col2, col3, col4 = st.columns(4)
    mostrarMetrica(col1, "Dataset local", "OK" if Path(rutaDatosLocal).exists() else "No")
    mostrarMetrica(col2, "ZIP Colab", "OK" if Path(rutaZipCudaEntrada).exists() else "Pendiente")
    mostrarMetrica(col3, "Metricas CPU", len(metricasCpu))
    mostrarMetrica(col4, "Metricas CUDA", len(metricasIntegracionCuda))

    if not metricasResumen.empty:
        resumen = metricasResumen.copy()
        resumen["tiempoSegundos"] = resumen["tiempoSegundos"].round(4)
        fig = px.bar(
            resumen,
            x="fase",
            y="tiempoSegundos",
            text="tiempoSegundos",
            color="fase",
            title="Tiempo real por fase",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(resumen, use_container_width=True)
    else:
        st.info("Ejecuta `python main.py --etapa resumen` cuando existan metricas.")

with tabDataset:
    st.subheader("Ingesta Dask DataFrame")
    if not metricasDask.empty:
        col1, col2, col3 = st.columns(3)
        mostrarMetrica(col1, "Imagenes", int(metricasDask["cantidadImagenes"].iloc[0]))
        mostrarMetrica(col2, "Clases", int(metricasDask["cantidadClases"].iloc[0]))
        mostrarMetrica(col3, "GB detectados", f"{metricasDask['tamanoTotalGb'].iloc[0]:.2f}")
        st.dataframe(metricasDask, use_container_width=True)
    else:
        st.warning("No existen metricas Dask.")

    if not metricasDaskClases.empty:
        fig = px.bar(
            metricasDaskClases,
            x="clase",
            y="tamanoTotalGb",
            color="clase",
            title="Tamano del dataset por clase",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(metricasDaskClases, use_container_width=True)

with tabCpu:
    st.subheader("Multiprocessing con OpenCV")
    if not metricasCpu.empty:
        clases = sorted(metricasCpu["clase"].dropna().unique())
        clasesSeleccionadas = st.multiselect("Clases", clases, default=clases)
        datos = metricasCpu[metricasCpu["clase"].isin(clasesSeleccionadas)].copy()
        datosOk = datos[datos["estado"] == "ok"].copy()

        col1, col2, col3, col4 = st.columns(4)
        mostrarMetrica(col1, "Imagenes OK", len(datosOk))
        mostrarMetrica(col2, "Procesos", int(datos["procesosUsados"].iloc[0]) if "procesosUsados" in datos and not datos.empty else 0)
        mostrarMetrica(col3, "Chunks", int(datos["cantidadChunks"].iloc[0]) if "cantidadChunks" in datos and not datos.empty else 0)
        mostrarMetrica(col4, "Tiempo real CPU", f"{datos['tiempoGlobalSegundos'].iloc[0]:.2f} s" if "tiempoGlobalSegundos" in datos and not datos.empty else "0 s")

        if not datosOk.empty:
            figDim = px.scatter(
                datosOk,
                x="anchoOriginal",
                y="altoOriginal",
                color="clase",
                title="Dimensiones originales",
                hover_data=["nombreArchivo"] if "nombreArchivo" in datosOk.columns else None,
            )
            st.plotly_chart(figDim, use_container_width=True)

            figTam = px.box(
                datosOk,
                x="clase",
                y="tamanoOriginalBytes",
                color="clase",
                title="Tamano original en bytes por clase",
            )
            figTam.update_layout(showlegend=False)
            st.plotly_chart(figTam, use_container_width=True)

            figMetricas = px.scatter(
                datosOk,
                x="media",
                y="desviacion",
                color="clase",
                title="Media vs desviacion despues del resize",
            )
            st.plotly_chart(figMetricas, use_container_width=True)

        if not metricasCpuClases.empty:
            st.markdown("#### Resumen por clase")
            st.dataframe(metricasCpuClases, use_container_width=True)

        st.markdown("#### Detalle por imagen")
        columnas = [
            "idImagen", "clase", "altoOriginal", "anchoOriginal", "canalesOriginales",
            "tamanoOriginalBytes", "altoResize", "anchoResize", "tamanoResizeBytes",
            "media", "desviacion", "tiempoSegundos", "estado", "error",
        ]
        columnas = [columna for columna in columnas if columna in datos.columns]
        st.dataframe(datos[columnas], use_container_width=True)
    else:
        st.warning("No existe metricasCpu.csv.")

with tabCuda:
    st.subheader("CUDA Laplaciano")
    if not metricasPrepararCuda.empty:
        col1, col2, col3 = st.columns(3)
        mostrarMetrica(col1, "RAW preparados", len(metricasPrepararCuda[metricasPrepararCuda["estado"] == "ok"]))
        mostrarMetrica(col2, "Alto", int(metricasPrepararCuda["alto"].max()))
        mostrarMetrica(col3, "Ancho", int(metricasPrepararCuda["ancho"].max()))

    if not metricasCudaColab.empty:
        st.markdown("#### Ejecucion GPU en Colab")
        col1, col2, col3 = st.columns(3)
        mostrarMetrica(col1, "Imagenes GPU", len(metricasCudaColab))
        if "tiempoKernelMs" in metricasCudaColab.columns:
            mostrarMetrica(col2, "Kernel promedio", f"{metricasCudaColab['tiempoKernelMs'].mean():.4f} ms")
        if "tiempoTotalGpuSegundos" in metricasCudaColab.columns:
            mostrarMetrica(col3, "Tiempo GPU total", f"{metricasCudaColab['tiempoTotalGpuSegundos'].sum():.2f} s")
        st.dataframe(metricasCudaColab, use_container_width=True)
    else:
        st.info("Aun no hay metricasCudaColab.csv. Ejecuta el notebook en Colab.")

    if not metricasIntegracionCuda.empty:
        st.markdown("#### Resultados integrados")
        clasesCuda = sorted(metricasIntegracionCuda["clase"].dropna().unique())
        claseSeleccionada = st.selectbox("Clase CUDA", clasesCuda)
        datosCuda = metricasIntegracionCuda[metricasIntegracionCuda["clase"] == claseSeleccionada]
        col1, col2, col3 = st.columns(3)
        mostrarMetrica(col1, "Imagenes integradas", len(datosCuda))
        mostrarMetrica(col2, "Media laplaciano", f"{datosCuda['mediaCuda'].mean():.4f}")
        mostrarMetrica(col3, "Desviacion laplaciano", f"{datosCuda['desviacionCuda'].mean():.4f}")
        fig = px.histogram(datosCuda, x="mediaCuda", nbins=40, title="Distribucion de respuesta laplaciana")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(datosCuda, use_container_width=True)

with tabImagenes:
    st.subheader("Comparacion original vs laplaciano")

    fuenteImagenes = metricasIntegracionCuda
    if fuenteImagenes.empty and not metricasPrepararCuda.empty:
        fuenteImagenes = metricasPrepararCuda

    if fuenteImagenes.empty:
        st.info("No hay resultados CUDA para comparar. Ejecuta Colab e integra el zip.")
    else:
        clasesImagenes = sorted(fuenteImagenes["clase"].dropna().unique())
        clase = st.selectbox("Clase para comparar", clasesImagenes)
        datosClase = fuenteImagenes[fuenteImagenes["clase"] == clase].copy()
        idsDisponibles = sorted(datosClase["idImagen"].astype(int).unique())

        if not idsDisponibles:
            st.info("No hay imagenes disponibles para la clase seleccionada.")
        else:
            idImagen = st.selectbox("Imagen", idsDisponibles)
            rutaOriginal = obtenerRutaOriginal(int(idImagen))
            tipoSalida, rutaSalida = obtenerSalidaLaplaciano(int(idImagen), clase)

            colOriginal, colLaplaciano = st.columns(2)
            with colOriginal:
                st.markdown("#### Original")
                if rutaOriginal:
                    st.image(str(rutaOriginal), caption=str(rutaOriginal), use_container_width=True)
                else:
                    st.warning("No se encontro la imagen original en la ruta guardada.")

            with colLaplaciano:
                st.markdown("#### Laplaciano CUDA")
                if tipoSalida == "png":
                    st.image(str(rutaSalida), caption=str(rutaSalida), use_container_width=True)
                elif tipoSalida == "npy":
                    st.image(cargarImagenNpy(rutaSalida), caption=str(rutaSalida), use_container_width=True)
                else:
                    st.warning("No hay PNG ni NPY de CUDA para esta imagen.")

            if not metricasIntegracionCuda.empty:
                filaMetricas = metricasIntegracionCuda[metricasIntegracionCuda["idImagen"] == int(idImagen)]
                if not filaMetricas.empty:
                    st.markdown("#### Metricas de la imagen")
                    st.dataframe(filaMetricas, use_container_width=True)
