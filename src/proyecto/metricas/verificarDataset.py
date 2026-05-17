from pathlib import Path
import pandas as pd
from proyecto.configuracion.configuracionProyecto import rutaDatosLocal, extensionesPermitidas, clasesEsperadas


def bytesAGb(valor):
    return valor / (1024 ** 3)


def main():
    if not rutaDatosLocal.exists():
        raise FileNotFoundError(f'No existe rutaDatosLocal: {rutaDatosLocal}')

    registros = []
    totalBytes = 0
    totalImagenes = 0

    print('Ruta dataset:', rutaDatosLocal)
    print('Carpetas detectadas:')
    for carpeta in sorted([p for p in rutaDatosLocal.iterdir() if p.is_dir()]):
        print(' -', carpeta.name)

    for clase in sorted([p for p in rutaDatosLocal.iterdir() if p.is_dir()]):
        imagenes = []
        for extension in extensionesPermitidas:
            imagenes.extend(clase.rglob(f'*{extension}'))
            imagenes.extend(clase.rglob(f'*{extension.upper()}'))
        imagenes = sorted(set(imagenes))
        tamanoClase = sum(p.stat().st_size for p in imagenes if p.exists())
        totalBytes += tamanoClase
        totalImagenes += len(imagenes)
        registros.append({
            'clase': clase.name,
            'cantidadImagenes': len(imagenes),
            'tamanoGb': bytesAGb(tamanoClase),
        })

    tabla = pd.DataFrame(registros)
    print('\nResumen por clase:')
    print(tabla.to_string(index=False))
    print('\nTotal imagenes:', totalImagenes)
    print('Tamano total GB:', round(bytesAGb(totalBytes), 4))

    if bytesAGb(totalBytes) < 5:
        print('\nADVERTENCIA: El dataset detectado no llega a 5 GB.')
        print('El enunciado pide un dataset >= 5 GB. Puedes usar este para pruebas,')
        print('pero para la entrega final conviene reemplazarlo o justificarlo con el docente.')
    else:
        print('\nOK: El dataset cumple el requisito de 5 GB o mas.')

    faltantes = [clase for clase in clasesEsperadas if not (rutaDatosLocal / clase).exists()]
    if faltantes:
        print('\nClases esperadas faltantes:', faltantes)


if __name__ == '__main__':
    main()
