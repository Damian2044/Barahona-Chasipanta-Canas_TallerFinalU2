import argparse
import sys
from pathlib import Path

RUTA_SRC = Path(__file__).resolve().parent / "src"
if str(RUTA_SRC) not in sys.path:
    sys.path.insert(0, str(RUTA_SRC))

ETAPAS_VALIDAS = ["todo", "descarga", "dask", "cpu", "zip-cuda", "integrar-cuda", "resumen"]


def construirArgumentos():
    parser = argparse.ArgumentParser(
        description="Pipeline hibrido: Dask, Multiprocessing, CUDA y Streamlit."
    )
    parser.add_argument(
        "--etapa",
        choices=ETAPAS_VALIDAS,
        default="todo",
        help="Etapa a ejecutar. Por defecto corre hasta dejar el ZIP listo para Colab.",
    )
    parser.add_argument(
        "--procesos",
        type=int,
        default=0,
        help="Cantidad de procesos CPU. 0 usa todos los disponibles.",
    )
    parser.add_argument(
        "--no-borrar",
        action="store_true",
        help="Conserva resultados anteriores. Sin esta opcion se regeneran salidas en etapa todo.",
    )
    parser.add_argument(
        "--zip-resultados",
        default="",
        help="Ruta opcional del zip de resultados CUDA descargado desde Colab.",
    )
    return parser.parse_args()


def ejecutarTodo(procesos, noBorrar):
    from proyecto.datos.descargarDataset import descargarDatasetSiHaceFalta
    from proyecto.fases.dask import ejecutarFaseDask
    from proyecto.fases.multiprocessingCpu import ejecutarFaseMultiprocessing
    from proyecto.metricas.crearResumenMetricas import crearMetricasResumen
    from proyecto.utilidades.archivos import crearCarpetasProyecto, limpiarResultados

    if not noBorrar:
        limpiarResultados()
    else:
        crearCarpetasProyecto()

    descargarDatasetSiHaceFalta()
    ejecutarFaseDask()
    ejecutarFaseMultiprocessing(procesos=procesos, crearZip=True)
    crearMetricasResumen()

    print("\nPipeline local completado.")
    print("Sube salidas/paquetes/paqueteCudaColab.zip a Colab.")
    print("Despues descarga el zip CUDA en salidas/paquetes/resultadosCuda.zip.")


def main():
    args = construirArgumentos()

    from proyecto.datos.descargarDataset import descargarDatasetSiHaceFalta
    from proyecto.fases.dask import ejecutarFaseDask
    from proyecto.fases.integrarCuda import ejecutarIntegracionCuda
    from proyecto.fases.multiprocessingCpu import ejecutarFaseMultiprocessing
    from proyecto.metricas.crearResumenMetricas import crearMetricasResumen
    from proyecto.utilidades.archivos import crearCarpetasProyecto, crearZipCudaEntrada

    crearCarpetasProyecto()

    if args.etapa == "todo":
        ejecutarTodo(args.procesos, args.no_borrar)
    elif args.etapa == "descarga":
        descargarDatasetSiHaceFalta()
    elif args.etapa == "dask":
        ejecutarFaseDask()
    elif args.etapa == "cpu":
        ejecutarFaseMultiprocessing(procesos=args.procesos, crearZip=True)
    elif args.etapa == "zip-cuda":
        rutaZip = crearZipCudaEntrada()
        print(f"ZIP listo para Colab: {rutaZip}")
    elif args.etapa == "integrar-cuda":
        ejecutarIntegracionCuda(args.zip_resultados or None)
        crearMetricasResumen()
    elif args.etapa == "resumen":
        crearMetricasResumen()


if __name__ == "__main__":
    main()
