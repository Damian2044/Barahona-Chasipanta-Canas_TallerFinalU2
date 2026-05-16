import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


from proyecto.configuracion.configuracionProyecto import (
    rutaCudaBin,
    rutaCudaNpy,
    rutaCudaPng,
    rutaMetricas,
    rutaMetricasIntegracionCuda,
    rutaMetricasPrepararCuda,
    rutaZipCudaResultados,
)
from proyecto.utilidades.archivos import descomprimirZipResultados


def leerMatrizCuda(rutaEntradaBin, alto, ancho):
    """Lee resultados CUDA nuevos uint8 y mantiene soporte para float32 antiguo."""
    cantidadElementos = alto * ancho
    tamanoArchivo = rutaEntradaBin.stat().st_size
    if tamanoArchivo == cantidadElementos:
        datos = np.fromfile(rutaEntradaBin, dtype=np.uint8)
        matrizVisual = datos.reshape((alto, ancho))
        matrizMetricas = matrizVisual.astype(np.float32) / 255.0
    else:
        datos = np.fromfile(rutaEntradaBin, dtype=np.float32)
        matrizMetricas = datos.reshape((alto, ancho))
        matrizVisual = np.clip(matrizMetricas * 255.0, 0, 255).astype(np.uint8)
    return matrizMetricas, matrizVisual


def integrarArchivoCuda(fila):
    """Convierte un binario CUDA en NPY, PNG y metricas para Streamlit."""
    inicio = time.perf_counter()
    idImagen = int(fila["idImagen"])
    clase = str(fila["clase"])
    alto = int(fila["alto"])
    ancho = int(fila["ancho"])
    rutaEntradaBin = rutaCudaBin / clase / f"imagen_{idImagen}_cuda.bin"

    try:
        matrizMetricas, matrizVisual = leerMatrizCuda(rutaEntradaBin, alto, ancho)

        rutaClaseNpy = rutaCudaNpy / clase
        rutaClasePng = rutaCudaPng / clase
        rutaClaseNpy.mkdir(parents=True, exist_ok=True)
        rutaClasePng.mkdir(parents=True, exist_ok=True)

        rutaSalidaNpy = rutaClaseNpy / f"imagen_{idImagen}_cuda.npy"
        rutaSalidaPng = rutaClasePng / f"imagen_{idImagen}_cuda.png"

        np.save(rutaSalidaNpy, matrizMetricas.astype(np.float32))
        cv2.imwrite(str(rutaSalidaPng), matrizVisual)

        tiempo = time.perf_counter() - inicio
        return {
            "idImagen": idImagen,
            "clase": clase,
            "rutaCudaBin": str(rutaEntradaBin),
            "rutaCudaNpy": str(rutaSalidaNpy),
            "rutaCudaPng": str(rutaSalidaPng),
            "alto": alto,
            "ancho": ancho,
            "mediaCuda": float(np.mean(matrizMetricas)),
            "desviacionCuda": float(np.std(matrizMetricas)),
            "minimoCuda": float(np.min(matrizMetricas)),
            "maximoCuda": float(np.max(matrizMetricas)),
            "tiempoSegundos": float(tiempo),
            "estado": "ok",
            "error": "",
        }
    except Exception as error:
        tiempo = time.perf_counter() - inicio
        return {
            "idImagen": idImagen,
            "clase": clase,
            "rutaCudaBin": str(rutaEntradaBin),
            "rutaCudaNpy": "",
            "rutaCudaPng": "",
            "alto": alto,
            "ancho": ancho,
            "mediaCuda": 0.0,
            "desviacionCuda": 0.0,
            "minimoCuda": 0.0,
            "maximoCuda": 0.0,
            "tiempoSegundos": float(tiempo),
            "estado": "error",
            "error": str(error),
        }


def ejecutarIntegracionCuda(rutaZip=None):
    """Descomprime el zip CUDA si existe e integra los binarios."""
    inicioGlobal = time.perf_counter()
    rutaCudaNpy.mkdir(parents=True, exist_ok=True)
    rutaCudaPng.mkdir(parents=True, exist_ok=True)
    rutaMetricas.mkdir(parents=True, exist_ok=True)

    rutaZip = Path(rutaZip) if rutaZip else rutaZipCudaResultados
    if rutaZip.exists():
        descomprimirZipResultados(rutaZip)

    if not rutaMetricasPrepararCuda.exists():
        raise FileNotFoundError("No existe metricasPrepararCuda.csv.")

    metricasPreparar = pd.read_csv(rutaMetricasPrepararCuda)
    metricasPreparar = metricasPreparar[metricasPreparar["estado"] == "ok"].copy()

    resultados = [
        integrarArchivoCuda(fila)
        for _, fila in tqdm(metricasPreparar.iterrows(), total=len(metricasPreparar), desc="Integrando CUDA")
    ]

    tiempoGlobal = time.perf_counter() - inicioGlobal
    metricasIntegracion = pd.DataFrame(resultados).sort_values("idImagen")
    metricasIntegracion["tiempoGlobalSegundos"] = float(tiempoGlobal)
    metricasIntegracion.to_csv(rutaMetricasIntegracionCuda, index=False)

    print("Integracion CUDA completada.")
    print(f"Tiempo total real: {tiempoGlobal:.2f} segundos")
    print(f"Metricas: {rutaMetricasIntegracionCuda}")
    return metricasIntegracion


if __name__ == "__main__":
    ejecutarIntegracionCuda()
