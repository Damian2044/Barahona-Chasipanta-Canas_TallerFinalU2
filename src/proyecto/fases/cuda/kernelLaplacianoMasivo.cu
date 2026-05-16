#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

// ###### CONFIGURACION CUDA ######
const int HILOS_X = 16;
const int HILOS_Y = 16;

// ###### ESTRUCTURA DE CADA IMAGEN ######
struct ImagenEntrada {
    int idImagen;
    string clase;
    string rutaRaw;
    int alto;
    int ancho;
};

// ###### KERNEL GPU: FILTRO LAPLACIANO CON MEMORIA COMPARTIDA ######
__global__ void aplicarLaplaciano(unsigned char* entrada, unsigned char* salida, int alto, int ancho) {
    __shared__ unsigned char tile[HILOS_Y + 2][HILOS_X + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    int lx = tx + 1;
    int ly = ty + 1;

    if (x < ancho && y < alto) {
        tile[ly][lx] = entrada[y * ancho + x];
    } else {
        tile[ly][lx] = 0;
    }

    if (tx == 0) {
        tile[ly][0] = (x > 0 && y < alto) ? entrada[y * ancho + (x - 1)] : 0;
    }
    if (tx == blockDim.x - 1) {
        tile[ly][HILOS_X + 1] = (x + 1 < ancho && y < alto) ? entrada[y * ancho + (x + 1)] : 0;
    }
    if (ty == 0) {
        tile[0][lx] = (y > 0 && x < ancho) ? entrada[(y - 1) * ancho + x] : 0;
    }
    if (ty == blockDim.y - 1) {
        tile[HILOS_Y + 1][lx] = (y + 1 < alto && x < ancho) ? entrada[(y + 1) * ancho + x] : 0;
    }

    __syncthreads();

    if (x >= ancho || y >= alto) {
        return;
    }

    int idx = y * ancho + x;
    if (x == 0 || y == 0 || x == ancho - 1 || y == alto - 1) {
        salida[idx] = 0;
        return;
    }

    int centro = tile[ly][lx] * 4;
    int vecinos = tile[ly][lx - 1] + tile[ly][lx + 1] + tile[ly - 1][lx] + tile[ly + 1][lx];
    int valor = abs(centro - vecinos);
    salida[idx] = (unsigned char)min(max(valor, 0), 255);
}

// ###### IO: LEER CSV SIMPLE GENERADO POR PYTHON ######
vector<string> separarCsv(const string& linea) {
    vector<string> columnas;
    string actual;
    for (char c : linea) {
        if (c == ',') {
            columnas.push_back(actual);
            actual.clear();
        } else {
            actual += c;
        }
    }
    columnas.push_back(actual);
    return columnas;
}

vector<ImagenEntrada> leerImagenesEntrada() {
    ifstream archivo("metricas/metricasPrepararCuda.csv");
    vector<ImagenEntrada> imagenes;
    string linea;

    getline(archivo, linea);
    while (getline(archivo, linea)) {
        vector<string> c = separarCsv(linea);
        if (c.size() >= 9 && c[8] == "ok") {
            imagenes.push_back({stoi(c[0]), c[1], c[2], stoi(c[3]), stoi(c[4])});
        }
    }
    return imagenes;
}

// ###### IO: ARCHIVOS BINARIOS RAW ######
void crearCarpetaSalida(const string& rutaArchivo) {
    size_t pos = rutaArchivo.find_last_of('/');
    if (pos != string::npos) {
        string comando = "mkdir -p \"" + rutaArchivo.substr(0, pos) + "\"";
        system(comando.c_str());
    }
}

void leerRaw(const string& ruta, unsigned char* datos, size_t bytes) {
    ifstream archivo(ruta, ios::binary);
    archivo.read((char*)datos, bytes);
    archivo.close();
}

void guardarRaw(const string& ruta, unsigned char* datos, size_t bytes) {
    crearCarpetaSalida(ruta);
    ofstream archivo(ruta, ios::binary);
    archivo.write((char*)datos, bytes);
    archivo.close();
}

// ###### GPU: RESERVAR MEMORIA, COPIAR, LANZAR KERNEL Y RETORNAR ######
float procesarEnGpu(unsigned char* hEntrada, unsigned char* hSalida, int alto, int ancho) {
    size_t bytes = alto * ancho * sizeof(unsigned char);

    unsigned char* dEntrada;
    unsigned char* dSalida;
    cudaMalloc(&dEntrada, bytes);
    cudaMalloc(&dSalida, bytes);

    cudaMemcpy(dEntrada, hEntrada, bytes, cudaMemcpyHostToDevice);

    dim3 hilos(HILOS_X, HILOS_Y);
    dim3 bloques((ancho + hilos.x - 1) / hilos.x, (alto + hilos.y - 1) / hilos.y);

    cudaEvent_t inicio, fin;
    cudaEventCreate(&inicio);
    cudaEventCreate(&fin);
    cudaEventRecord(inicio);

    aplicarLaplaciano<<<bloques, hilos>>>(dEntrada, dSalida, alto, ancho);
    cudaDeviceSynchronize();

    cudaEventRecord(fin);
    cudaEventSynchronize(fin);

    float tiempoKernelMs = 0.0f;
    cudaEventElapsedTime(&tiempoKernelMs, inicio, fin);

    cudaMemcpy(hSalida, dSalida, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dEntrada);
    cudaFree(dSalida);
    cudaEventDestroy(inicio);
    cudaEventDestroy(fin);

    return tiempoKernelMs;
}

// ###### MAIN: PROCESA TODAS LAS IMAGENES DEL PAQUETE ######
int main() {
    system("mkdir -p metricas salidas/cudaBin");

    vector<ImagenEntrada> imagenes = leerImagenesEntrada();
    ofstream metricas("metricas/metricasCudaColab.csv");
    metricas << "idImagen,clase,rutaRaw,rutaCudaBin,alto,ancho,tiempoKernelMs,tiempoTotalGpuSegundos,hilosX,hilosY,estado,error\n";

    for (ImagenEntrada img : imagenes) {
        auto inicioTotal = chrono::high_resolution_clock::now();
        string rutaSalida = "salidas/cudaBin/" + img.clase + "/imagen_" + to_string(img.idImagen) + "_cuda.bin";
        size_t bytes = img.alto * img.ancho * sizeof(unsigned char);

        unsigned char* hEntrada = (unsigned char*)malloc(bytes);
        unsigned char* hSalida = (unsigned char*)malloc(bytes);

        try {
            leerRaw(img.rutaRaw, hEntrada, bytes);
            float tiempoKernelMs = procesarEnGpu(hEntrada, hSalida, img.alto, img.ancho);
            guardarRaw(rutaSalida, hSalida, bytes);

            auto finTotal = chrono::high_resolution_clock::now();
            chrono::duration<double> tiempoTotal = finTotal - inicioTotal;

            metricas << img.idImagen << "," << img.clase << "," << img.rutaRaw << "," << rutaSalida
                     << "," << img.alto << "," << img.ancho << "," << tiempoKernelMs << ","
                     << tiempoTotal.count() << "," << HILOS_X << "," << HILOS_Y << ",ok,\n";
        } catch (...) {
            auto finTotal = chrono::high_resolution_clock::now();
            chrono::duration<double> tiempoTotal = finTotal - inicioTotal;

            metricas << img.idImagen << "," << img.clase << "," << img.rutaRaw << "," << rutaSalida
                     << "," << img.alto << "," << img.ancho << ",0," << tiempoTotal.count()
                     << "," << HILOS_X << "," << HILOS_Y << ",error,error en CUDA o lectura\n";
        }

        free(hEntrada);
        free(hSalida);
    }

    cout << "Procesamiento CUDA Laplaciano completado." << endl;
    return 0;
}
