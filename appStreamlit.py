from pathlib import Path
import sys

RUTA_SRC = Path(__file__).resolve().parent / "src"
if str(RUTA_SRC) not in sys.path:
    sys.path.insert(0, str(RUTA_SRC))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageOps

from proyecto.configuracion.configuracionProyecto import (
    enlaceDatasetKaggle,
    nombreDataset,
    rutaBaseProyecto,
    rutaDatosLocal,
    rutaMetricasCpu,
    rutaMetricasCpuClases,
    rutaMetricasCudaColab,
    rutaMetricasDask,
    rutaMetricasDaskClases,
    rutaMetricasPrepararCuda,
    rutaMetricasResumen,
    rutaTablaRutas,
    rutaZipCudaEntrada,
    rutaZipCudaResultados,
    tamanoObjetivo,
)

st.set_page_config(page_title="Procesamiento Híbrido de Imágenes", layout="wide")
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#60a5fa", "#34d399", "#fbbf24", "#f472b6", "#a78bfa"]

st.markdown(
    """
    <style>
    .stApp {background: #f6f8fb; color: #17202a;}
    .main .block-container {padding-top: 1.2rem; max-width: 1500px;}
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] * {color: #17202a;}
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: .75rem .9rem;
        box-shadow: 0 1px 12px rgba(15, 23, 42, .08);
    }
    div[data-testid="stMetric"] label {color: #64748b;}
    div[data-testid="stMetricValue"] {color: #0f172a;}
    div[data-testid="stTabs"] button p {font-size: 1rem; font-weight: 700;}
    div[data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }
    div[data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
    }
    .dash-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        margin: .55rem 0;
        box-shadow: 0 10px 24px rgba(15, 23, 42, .08);
    }
    .dash-card h3 {
        margin: 0 0 .35rem 0;
        color: #0f172a;
        font-size: 1.05rem;
    }
    .dash-card p {
        margin: 0;
        color: #475569;
        line-height: 1.45;
    }
    .side-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: .75rem;
        margin-bottom: .7rem;
    }
    .side-card b {color: #0f172a;}
    .side-card span {color: #475569; font-size: .9rem;}
    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 14px;
        background: #ffffff;
        color: #0f172a;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
        box-shadow: 0 14px 38px rgba(15, 23, 42, .08);
    }
    .hero h1 {margin: 0; font-size: 3.05rem; line-height: 1.05; letter-spacing: 0;}
    .hero p {margin: .55rem 0 0 0; color: #475569; font-size: 1.05rem;}
    .dataset-strip {
        display: flex;
        gap: .65rem;
        flex-wrap: wrap;
        margin: -.35rem 0 1rem 0;
    }
    .pill {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        color: #334155;
        border-radius: 999px;
        padding: .35rem .7rem;
        font-size: .9rem;
    }
    .note {
        padding: .85rem 1rem;
        border-left: 5px solid #2563eb;
        background: #ffffff;
        color: #334155;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        margin-bottom: .8rem;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        margin: .2rem 0 .6rem 0;
    }
    .mini {
        color: #475569;
        font-size: .95rem;
        margin-bottom: .8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero">
        <h1>Procesamiento a Gran Escala con Arquitectura Híbrida</h1>
        <p>DASK + Multiprocessing + NumPy + CUDA + Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def cargarCsvSeguro(ruta):
    ruta = Path(ruta)
    if ruta.exists():
        return pd.read_csv(ruta)
    return pd.DataFrame()


def mb(valor):
    return float(valor) / (1024 ** 2)


def gb(valor):
    return float(valor) / (1024 ** 3)


def mostrarMetrica(columna, titulo, valor, sufijo=""):
    if pd.isna(valor):
        valor = 0
    columna.metric(titulo, f"{valor}{sufijo}")


def bloqueTexto(titulo, texto):
    st.markdown(f'<div class="section-title">{titulo}</div><div class="mini">{texto}</div>', unsafe_allow_html=True)


def tarjeta(titulo, texto):
    st.markdown(f'<div class="dash-card"><h3>{titulo}</h3><p>{texto}</p></div>', unsafe_allow_html=True)


def tablaOpcional(titulo, dataframe, filas=None):
    if dataframe.empty:
        return
    with st.expander(titulo):
        datos = dataframe.head(filas) if filas else dataframe
        st.dataframe(datos, use_container_width=True)


def resolverRuta(ruta):
    if ruta is None or pd.isna(ruta) or str(ruta).strip() == "":
        return None
    ruta = Path(str(ruta))
    if ruta.exists():
        return ruta
    rutaLocal = rutaBaseProyecto / ruta
    if rutaLocal.exists():
        return rutaLocal
    return ruta


def obtenerRutaOriginal(idImagen):
    for tabla, columna in [(metricasCpu, "rutaOriginal"), (tablaRutas, "rutaImagen")]:
        if not tabla.empty and columna in tabla.columns:
            fila = tabla[tabla["idImagen"].astype(int) == int(idImagen)]
            if not fila.empty:
                ruta = resolverRuta(fila[columna].iloc[0])
                if ruta and ruta.exists():
                    return ruta
    return None


def obtenerRutaRaw(idImagen):
    if metricasPrepararCuda.empty or "rutaRaw" not in metricasPrepararCuda.columns:
        return None
    fila = metricasPrepararCuda[metricasPrepararCuda["idImagen"].astype(int) == int(idImagen)]
    if fila.empty:
        return None
    ruta = resolverRuta(fila["rutaRaw"].iloc[0])
    return ruta if ruta and ruta.exists() else None


def obtenerFilaCuda(idImagen):
    if metricasCudaColab.empty:
        return pd.Series(dtype=object)
    fila = metricasCudaColab[metricasCudaColab["idImagen"].astype(int) == int(idImagen)]
    if fila.empty:
        return pd.Series(dtype=object)
    return fila.iloc[0]


def obtenerRutaCudaBin(idImagen, clase):
    fila = obtenerFilaCuda(idImagen)
    if not fila.empty and "rutaCudaBin" in fila:
        ruta = resolverRuta(fila["rutaCudaBin"])
        if ruta and ruta.exists():
            return ruta
    rutaConvencion = rutaBaseProyecto / "salidas" / "cudaBin" / str(clase) / f"imagen_{int(idImagen)}_cuda.bin"
    return rutaConvencion if rutaConvencion.exists() else None


def leerBinarioImagen(ruta, alto=tamanoObjetivo, ancho=tamanoObjetivo):
    ruta = Path(ruta)
    cantidad = int(alto) * int(ancho)
    if ruta.stat().st_size == cantidad:
        datos = np.fromfile(ruta, dtype=np.uint8)
        return datos.reshape((int(alto), int(ancho)))
    datos = np.fromfile(ruta, dtype=np.float32)
    matriz = datos.reshape((int(alto), int(ancho)))
    return np.clip(matriz * 255.0, 0, 255).astype(np.uint8)


def imagenOriginalVisual(ruta):
    imagen = Image.open(ruta)
    imagen = ImageOps.exif_transpose(imagen.convert("RGB"))
    return ImageOps.contain(imagen, (320, 320), Image.Resampling.LANCZOS)


def matrizVisual(matriz):
    imagen = Image.fromarray(matriz).convert("RGB")
    return ImageOps.contain(imagen, (320, 320), Image.Resampling.LANCZOS)


def mostrarImagenPanel(columna, titulo, imagen, detalle):
    with columna:
        st.markdown(f"**{titulo}**")
        if imagen:
            st.image(imagen, use_container_width=True)
            st.caption(detalle)
        else:
            st.warning("No disponible")


def crearComparacion(imagenes):
    anchoPanel = 360
    altoPanel = 390
    lienzo = Image.new("RGB", (anchoPanel * len(imagenes), altoPanel), "white")
    for indice, (titulo, imagen) in enumerate(imagenes):
        panelX = indice * anchoPanel
        if imagen:
            x = panelX + (anchoPanel - imagen.width) // 2
            y = 30 + (360 - imagen.height) // 2
            lienzo.paste(imagen, (x, y))
        # Titulos sencillos para que la comparacion exportada se entienda.
        import PIL.ImageDraw as ImageDraw
        draw = ImageDraw.Draw(lienzo)
        draw.text((panelX + 12, 8), titulo, fill=(20, 20, 20))
    return lienzo


metricasDask = cargarCsvSeguro(rutaMetricasDask)
metricasDaskClases = cargarCsvSeguro(rutaMetricasDaskClases)
metricasCpu = cargarCsvSeguro(rutaMetricasCpu)
metricasCpuClases = cargarCsvSeguro(rutaMetricasCpuClases)
metricasPrepararCuda = cargarCsvSeguro(rutaMetricasPrepararCuda)
metricasCudaColab = cargarCsvSeguro(rutaMetricasCudaColab)
metricasResumen = cargarCsvSeguro(rutaMetricasResumen)
tablaRutas = cargarCsvSeguro(rutaTablaRutas)

with st.sidebar:
    st.markdown("## Resumen")
    st.caption("Indicadores principales del pipeline.")

    st.markdown(
        """
        <div class="side-card">
            <b>Dataset de hojas de maíz</b><br>
            <span>4000 imágenes RGB tomadas en campo. El dataset tiene 4 clases balanceadas, con 1000 imágenes por clase.</span>
            <br><br>
            <span>• Daun Sehat: hoja saludable, sin enfermedad visible.</span><br>
            <span>• Bercak Daun: hoja con manchas o puntos de enfermedad.</span><br>
            <span>• Hawar Daun: hoja con zonas secas, quemadas o dañadas.</span><br>
            <span>• Karat Daun: hoja con roya, manchas tipo óxido causadas por hongos.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not metricasDask.empty:
        filaDask = metricasDask.iloc[0]
        st.markdown(
            f'<div class="side-card"><b>🧩 DASK</b><br><span>{int(filaDask["cantidadImagenes"])} imágenes | {float(filaDask["tamanoTotalGb"]):.2f} GB | {float(filaDask["tiempoSegundos"]):.2f} s</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("DASK pendiente")

    if not metricasCpu.empty:
        cpuOk = metricasCpu[metricasCpu["estado"] == "ok"] if "estado" in metricasCpu.columns else metricasCpu
        procesos = int(metricasCpu["procesosUsados"].iloc[0]) if "procesosUsados" in metricasCpu.columns else 0
        tiempoCpu = float(metricasCpu["tiempoGlobalSegundos"].iloc[0]) if "tiempoGlobalSegundos" in metricasCpu.columns else 0
        st.markdown(
            f'<div class="side-card"><b>⚙️ CPU</b><br><span>{len(cpuOk)} imágenes | {procesos} procesos | {tiempoCpu:.2f} s</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("CPU pendiente")

    if not metricasCudaColab.empty:
        kernelProm = metricasCudaColab["tiempoKernelMs"].mean() if "tiempoKernelMs" in metricasCudaColab.columns else 0
        tiempoCuda = metricasCudaColab["tiempoTotalGpuSegundos"].sum() if "tiempoTotalGpuSegundos" in metricasCudaColab.columns else 0
        st.markdown(
            f'<div class="side-card"><b>🚀 CUDA</b><br><span>{len(metricasCudaColab)} imágenes | {kernelProm:.4f} ms kernel | {tiempoCuda:.2f} s</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("CUDA pendiente")

st.markdown(
    f"""
    <div class="dataset-strip">
        <span class="pill">Dataset: {nombreDataset}</span>
        <span class="pill">Kaggle: {enlaceDatasetKaggle}</span>
        <span class="pill">Matriz: {tamanoObjetivo}x{tamanoObjetivo}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

tabResumen, tabDask, tabCpu, tabCuda, tabImagenes = st.tabs([
    "📊 Resumen",
    "🧩 DASK",
    "⚙️ CPU",
    "🚀 CUDA",
    "🖼️ Imágenes",
])

with tabResumen:
    if not metricasResumen.empty:
        resumen = metricasResumen.copy()
        resumen["tiempoSegundos"] = resumen["tiempoSegundos"].round(4)
        total = resumen["tiempoSegundos"].sum()
        resumen["porcentajeTiempo"] = (resumen["tiempoSegundos"] / total * 100).round(2) if total else 0

        bloqueTexto(
            "Tiempos finales",
            "Comparación directa de las tres fases computacionales medidas: DASK para metadatos, CPU para preparar matrices y CUDA para aplicar el filtro de enfoque.",
        )
        colsTiempo = st.columns(len(resumen))
        iconos = ["▦", "⚙", "▣"]
        for indice, (_, fila) in enumerate(resumen.iterrows()):
            mostrarMetrica(
                colsTiempo[indice],
                f"{iconos[indice % len(iconos)]} {fila['fase']}",
                f"{float(fila['tiempoSegundos']):.2f} s",
            )

        colBarra, colPastel = st.columns([1.35, 1])
        with colBarra:
            fig = px.bar(
                resumen,
                x="fase",
                y="tiempoSegundos",
                text="tiempoSegundos",
                color="fase",
                title="Tiempo neto por fase computacional",
            )
            fig.update_layout(showlegend=False, yaxis_title="Segundos", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        with colPastel:
            figPie = px.pie(
                resumen,
                names="fase",
                values="tiempoSegundos",
                title="Porcentaje del tiempo total",
                hole=0.42,
            )
            st.plotly_chart(figPie, use_container_width=True)

        tablaOpcional("Ver tabla de tiempos finales", resumen)
    else:
        st.info("Ejecuta `python main.py --etapa resumen` cuando existan métricas.")

with tabDask:
    st.markdown('<div class="note"><b>▦ Fase DASK.</b> Recorre el dataset como metadatos de archivos: ruta, nombre, clase, extensión y tamaño. Con DASK se construyen particiones de forma perezosa, se limpian registros inválidos, se tipan columnas y se agrupa por clase para resumir el volumen del dataset.</div>', unsafe_allow_html=True)
    if not metricasDask.empty:
        fila = metricasDask.iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        mostrarMetrica(col1, "Imágenes detectadas", int(fila["cantidadImagenes"]))
        mostrarMetrica(col2, "Clases", int(fila["cantidadClases"]))
        mostrarMetrica(col3, "Tamaño total", f"{float(fila['tamanoTotalGb']):.2f} GB")
        mostrarMetrica(col4, "Tiempo DASK", f"{float(fila['tiempoSegundos']):.2f} s")
        tarjeta(
            "Operación realizada",
            "DASK generó una tabla de metadatos por imagen y luego agrupó por clase para medir volumen, cantidad y tamaños sin abrir cada imagen como matriz.",
        )
    else:
        st.warning("No existen métricas de DASK.")

    if not metricasDaskClases.empty:
        clasesDask = metricasDaskClases.copy()
        clasesDask["Tamaño total (GB)"] = clasesDask["tamanoTotalGb"].round(4)
        clasesDask["Tamaño promedio (MB)"] = clasesDask["tamanoPromedioBytes"].map(mb).round(3)
        clasesDask["Tamaño mínimo (MB)"] = clasesDask["tamanoMinimoBytes"].map(mb).round(3)
        clasesDask["Tamaño máximo (MB)"] = clasesDask["tamanoMaximoBytes"].map(mb).round(3)

        fig = px.bar(
            clasesDask,
            x="clase",
            y="Tamaño total (GB)",
            color="clase",
            text="Tamaño total (GB)",
            title="Volumen del dataset por clase",
        )
        fig.update_layout(showlegend=False, xaxis_title="Clase", yaxis_title="GB")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Resumen DASK por clase")
        tablaDaskClase = (
            clasesDask[[
                "clase", "cantidadImagenes", "Tamaño total (GB)",
                "Tamaño promedio (MB)", "Tamaño mínimo (MB)", "Tamaño máximo (MB)",
            ]]
        )
        tablaOpcional("Ver resumen DASK por clase", tablaDaskClase)

    if not tablaRutas.empty:
        tablaOpcional(f"Ver tabla generada por DASK ({len(tablaRutas)} filas)", tablaRutas, filas=500)

with tabCpu:
    st.markdown('<div class="note"><b>⚙ Fase CPU.</b> Usa <b>multiprocessing.Pool</b> para repartir los chunks entre varios procesos y los recorre con <code>imap_unordered</code>. Se usa esta variante porque entrega cada resultado apenas un proceso termina su chunk, sin esperar el orden original de la lista; así se aprovechan mejor los núcleos cuando algunas imágenes tardan más que otras. Cada chunk llama a <code>procesarImagenCpu</code>, que lee con OpenCV, convierte a escala de grises, redimensiona a 512x512, crea una matriz NumPy normalizada, calcula métricas y guarda un RAW uint8 para CUDA.</div>', unsafe_allow_html=True)
    if metricasCpu.empty:
        st.warning("No existe el archivo metricasCpu.csv.")
    else:
        clases = sorted(metricasCpu["clase"].dropna().unique())
        clasesSeleccionadas = st.multiselect("Clases", clases, default=clases)
        datos = metricasCpu[metricasCpu["clase"].isin(clasesSeleccionadas)].copy()
        datosOk = datos[datos["estado"] == "ok"].copy()

        col1, col2, col3, col4 = st.columns(4)
        mostrarMetrica(col1, "Imágenes OK", len(datosOk))
        mostrarMetrica(col2, "Procesos", int(datos["procesosUsados"].iloc[0]) if "procesosUsados" in datos and not datos.empty else 0)
        mostrarMetrica(col3, "Chunks", int(datos["cantidadChunks"].iloc[0]) if "cantidadChunks" in datos and not datos.empty else 0)
        mostrarMetrica(col4, "Tiempo CPU", f"{datos['tiempoGlobalSegundos'].iloc[0]:.2f} s" if "tiempoGlobalSegundos" in datos and not datos.empty else "0 s")

        if not datos.empty and {"procesosUsados", "cantidadChunks", "tamanoChunk"}.issubset(datos.columns):
            procesos = int(datos["procesosUsados"].iloc[0])
            chunks = int(datos["cantidadChunks"].iloc[0])
            tamanoChunk = int(datos["tamanoChunk"].iloc[0])
            st.caption(
                f"El trabajo se divide en {chunks} chunks para {procesos} procesos. "
                f"Cada chunk contiene hasta {tamanoChunk} imágenes. Por defecto, el pipeline usa la cantidad máxima de CPU disponible."
            )

        if not datos.empty and "indiceChunk" in datos.columns:
            tablaChunks = (
                datos
                .groupby("indiceChunk", dropna=False)
                .size()
                .reset_index(name="tamanoChunkReal")
                .sort_values("indiceChunk")
            )
            tablaChunks["chunk"] = tablaChunks["indiceChunk"].astype(int) + 1
            tablaChunks = tablaChunks[["chunk", "tamanoChunkReal"]].rename(
                columns={
                    "chunk": "Chunk",
                    "tamanoChunkReal": "Tamano del chunk",
                }
            )
            with st.expander("Ver tabla de chunks CPU", expanded=False):
                st.dataframe(tablaChunks, use_container_width=True, hide_index=True)

        if not datosOk.empty:
            datosOk["tamanoOriginalMb"] = datosOk["tamanoOriginalBytes"].map(mb)
            datosOk["tamanoResizeKb"] = datosOk["tamanoResizeBytes"] / 1024
            colGraf1, colGraf2 = st.columns(2)
            with colGraf1:
                dimensiones = (
                    datosOk
                    .groupby(["anchoOriginal", "altoOriginal", "clase"])
                    .size()
                    .reset_index(name="cantidadImagenes")
                )
                figDim = px.scatter(
                    dimensiones,
                    x="anchoOriginal",
                    y="altoOriginal",
                    size="cantidadImagenes",
                    color="clase",
                    title="Dimensiones originales agrupadas",
                    hover_data=["cantidadImagenes"],
                    size_max=42,
                )
                figDim.update_layout(xaxis_title="Ancho original (px)", yaxis_title="Alto original (px)")
                st.plotly_chart(figDim, use_container_width=True)
            with colGraf2:
                figTam = px.box(
                    datosOk,
                    x="clase",
                    y="tamanoOriginalMb",
                    color="clase",
                    title="Tamaño original por clase (MB)",
                )
                figTam.update_layout(showlegend=False, yaxis_title="MB")
                st.plotly_chart(figTam, use_container_width=True)

            figMetricas = px.scatter(
                datosOk,
                x="media",
                y="desviacion",
                color="clase",
                title="Brillo promedio vs contraste después del redimensionado",
                hover_data=["idImagen", "tamanoResizeKb"],
            )
            st.plotly_chart(figMetricas, use_container_width=True)

        if not metricasCpuClases.empty:
            resumenCpu = metricasCpuClases.copy()
            resumenCpu["altoOriginalPromedio"] = resumenCpu["altoOriginalPromedio"].round(2)
            resumenCpu["anchoOriginalPromedio"] = resumenCpu["anchoOriginalPromedio"].round(2)
            resumenCpu["tamanoOriginalMbPromedio"] = resumenCpu["tamanoOriginalMbPromedio"].round(3)
            resumenCpu["tamanoResizeKbPromedio"] = resumenCpu["tamanoResizeKbPromedio"].round(3)
            resumenCpu["mediaPromedio"] = resumenCpu["mediaPromedio"].round(4)
            resumenCpu["desviacionPromedio"] = resumenCpu["desviacionPromedio"].round(4)
            resumenCpu["tiempoPromedioSegundos"] = resumenCpu["tiempoPromedioSegundos"].round(4)
            tablaOpcional("Ver resumen CPU por clase", resumenCpu)

        columnas = [
            "idImagen", "clase", "altoOriginal", "anchoOriginal", "canalesOriginales",
            "tamanoOriginalBytes", "altoResize", "anchoResize", "tamanoResizeBytes",
            "media", "desviacion", "tiempoSegundos", "estado", "error",
        ]
        columnas = [columna for columna in columnas if columna in datos.columns]
        tablaOpcional(f"Ver detalle CPU por imagen ({len(datos)} filas)", datos[columnas], filas=1000)

with tabCuda:
    st.markdown('<div class="note"><b>▣ Fase CUDA.</b> Recibe los RAW 512x512 generados por CPU y ejecuta <code>aplicarEnfoque</code>, una convolución 3x3 con memoria compartida para enfoque. Cada hilo procesa un píxel usando el bloque compartido cargado en GPU.</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    mostrarMetrica(col1, "RAW preparados", len(metricasPrepararCuda[metricasPrepararCuda["estado"] == "ok"]) if not metricasPrepararCuda.empty and "estado" in metricasPrepararCuda else len(metricasPrepararCuda))
    mostrarMetrica(col2, "ZIP entrada", "OK" if Path(rutaZipCudaEntrada).exists() else "Pendiente")
    mostrarMetrica(col3, "ZIP resultados", "OK" if Path(rutaZipCudaResultados).exists() else "Pendiente")
    mostrarMetrica(col4, "Métricas GPU", len(metricasCudaColab))

    if metricasCudaColab.empty:
        st.info("Aún no hay metricasCudaColab.csv. Ejecuta Colab y luego `python main.py --etapa extraer-cuda --zip-resultados src/salidas/paquetes/resultadosCuda.zip`.")
    else:
        cuda = metricasCudaColab.copy()
        colGpu1, colGpu2, colGpu3, colGpu4 = st.columns(4)
        mostrarMetrica(colGpu1, "Imágenes procesadas", len(cuda))
        mostrarMetrica(colGpu2, "Kernel promedio", f"{cuda['tiempoKernelMs'].mean():.4f} ms" if "tiempoKernelMs" in cuda else "0 ms")
        mostrarMetrica(colGpu3, "Tiempo total CUDA", f"{cuda['tiempoTotalGpuSegundos'].sum():.2f} s" if "tiempoTotalGpuSegundos" in cuda else "0 s")
        mostrarMetrica(colGpu4, "Bloque CUDA", f"{int(cuda['hilosX'].iloc[0])}x{int(cuda['hilosY'].iloc[0])}" if {"hilosX", "hilosY"}.issubset(cuda.columns) else "N/D")

        tablaOpcional("Ver métricas generadas por CUDA", cuda, filas=1000)
        st.caption("tiempoKernelMs mide solo la ejecución del kernel. tiempoTotalGpuSegundos incluye lectura, copias CPU/GPU, kernel y escritura del binario.")

with tabImagenes:
    st.markdown('<div class="note"><b>Comparación visual.</b> Selecciona clases y cantidad de muestras. La vista compara la imagen original, el RAW preparado por CPU y la salida CUDA en enfoque.</div>', unsafe_allow_html=True)

    fuente = metricasCudaColab if not metricasCudaColab.empty else metricasPrepararCuda
    if fuente.empty:
        st.info("No hay imágenes listas. Ejecuta CPU y luego extrae los resultados CUDA.")
    else:
        clasesImagenes = sorted(fuente["clase"].dropna().unique())
        colClase, colCantidad = st.columns([2, 1])
        clasesSeleccionadas = colClase.multiselect(
            "Clases",
            clasesImagenes,
            default=clasesImagenes[:1] if clasesImagenes else [],
        )
        datosGaleria = fuente[fuente["clase"].isin(clasesSeleccionadas)].copy()
        if "estado" in datosGaleria.columns:
            datosGaleria = datosGaleria[datosGaleria["estado"] == "ok"]

        if datosGaleria.empty:
            st.info("No hay imágenes con los filtros seleccionados.")
        else:
            datosGaleria["idImagen"] = datosGaleria["idImagen"].astype(int)
            datosGaleria = datosGaleria.sort_values(["clase", "idImagen"])
            cantidad = colCantidad.slider(
                "Imágenes por página",
                min_value=1,
                max_value=min(12, int(len(datosGaleria))),
                value=min(3, int(len(datosGaleria))),
                step=1,
            )

            totalImagenes = int(len(datosGaleria))
            totalPaginas = max(1, int(np.ceil(totalImagenes / int(cantidad))))
            pagina = st.slider("Página", min_value=1, max_value=totalPaginas, value=1, step=1)
            inicio = (int(pagina) - 1) * int(cantidad)
            fin = inicio + int(cantidad)
            datosMostrar = datosGaleria.iloc[inicio:fin]
            st.caption(
                f"Mostrando {inicio + 1}-{min(fin, totalImagenes)} de {totalImagenes} imágenes filtradas."
            )

            if datosMostrar.empty:
                st.info("No hay imágenes para mostrar en esta página.")

            for _, fila in datosMostrar.iterrows():
                idImagen = int(fila["idImagen"])
                clase = str(fila["clase"])
                rutaOriginal = obtenerRutaOriginal(idImagen)
                rutaRaw = obtenerRutaRaw(idImagen)
                rutaCuda = obtenerRutaCudaBin(idImagen, clase)

                original = imagenOriginalVisual(rutaOriginal) if rutaOriginal else None
                raw = matrizVisual(leerBinarioImagen(rutaRaw)) if rutaRaw else None
                cuda = matrizVisual(leerBinarioImagen(rutaCuda)) if rutaCuda else None

                with st.container(border=True):
                    st.markdown(f"**Imagen {idImagen} | {clase}**")
                    colOriginal, colRaw, colCuda = st.columns(3)
                    mostrarImagenPanel(colOriginal, "Original", original, "Archivo del dataset")
                    mostrarImagenPanel(colRaw, "RAW antes de CUDA", raw, "Gris 512x512 preparado por CPU")
                    mostrarImagenPanel(colCuda, "CUDA enfoque", cuda, "BIN generado por GPU")
