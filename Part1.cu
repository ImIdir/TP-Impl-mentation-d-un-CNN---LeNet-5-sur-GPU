#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

// Initialisation d'une matrice sur CPU
void MatrixInit(float *M, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            M[i * p + j] = ((float)rand() / RAND_MAX) * 2 - 1;
        }
    }
}

// Affichage d'une matrice sur CPU
void MatrixPrint(float *M, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            printf("%f ", M[i * p + j]);
        }
        printf("\n");
    }
}

// Addition de deux matrices sur CPU
void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            Mout[i * p + j] = M1[i * p + j] + M2[i * p + j];
        }
    }
}

// Addition de deux matrices sur GPU
__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < p) {
        Mout[row * p + col] = M1[row * p + col] + M2[row * p + col];
    }
}

// Multiplication de deux matrices sur CPU
void MatrixMult(float *M1, float *M2, float *Mout, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Mout[i * n + j] = 0;
            for (int k = 0; k < n; k++) {
                Mout[i * n + j] += M1[i * n + k] * M2[k * n + j];
            }
        }
    }
}

// Multiplication de deux matrices sur GPU
__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        float value = 0;
        for (int k = 0; k < n; k++) {
            value += M1[row * n + k] * M2[k * n + col];
        }
        Mout[row * n + col] = value;
    }
}

// Mesure du temps GPU
double measureGPUTimeadd(dim3 grid, dim3 block, float *d_M1, float *d_M2, float *d_Mout, int n, int p) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaMatrixAdd<<<grid, block>>>(d_M1, d_M2, d_Mout, n, p);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    return milliseconds / 1000.0;
}

double measureGPUTimemult(dim3 grid, dim3 block, float *d_M1, float *d_M2, float *d_Mout, int n) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaMatrixMult<<<grid, block>>>(d_M1, d_M2, d_Mout, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    return milliseconds / 1000.0;
}


int main(int argc, char *argv[]) {
    int n = 100, p = 100; // Dimensions des matrices par défaut
    if (argc == 3) {
        n = atoi(argv[1]);
        p = atoi(argv[2]);
    }

    // Allocation mémoire CPU
    float *M1 = (float*)malloc(n * p * sizeof(float));
    float *M2 = (float*)malloc(n * p * sizeof(float));
    float *Mout = (float*)malloc(n * p * sizeof(float));

    // Allocation mémoire GPU
    float *d_M1, *d_M2, *d_Mout;
    size_t size = n * p * sizeof(float);
    cudaMalloc((void**)&d_M1, size);
    cudaMalloc((void**)&d_M2, size);
    cudaMalloc((void**)&d_Mout, size);

    // Initialisation des matrices CPU
    srand(time(NULL));
    MatrixInit(M1, n, p);
    MatrixInit(M2, n, p);

    // Copie des données du CPU vers le GPU
    cudaMemcpy(d_M1, M1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, M2, size, cudaMemcpyHostToDevice);

    // Paramètres CUDA
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (p + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Mesure du temps CPU
    clock_t start = clock();
    MatrixAdd(M1, M2, Mout, n, p);
    clock_t end = clock();
    double cpuTime = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Temps CPU (Addition): %f secondes\n", cpuTime);

    // Mesure du temps CPU pour la multiplication 
    start = clock(); 
    MatrixMult(M1, M1, Mout, n); 
    end = clock(); 
    double cpuTimeMult = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Temps CPU (Multiplication): %f secondes\n", cpuTimeMult); 

    // Mesure du temps GPU
    double gpuTime = measureGPUTimeadd(blocksPerGrid, threadsPerBlock, d_M1, d_M2, d_Mout, n, p);
    printf("Temps GPU (Addition): %f secondes\n", gpuTime);

    // Mesure du temps GPU pour la multiplication 
    double gpuTimeMult = measureGPUTimemult(blocksPerGrid, threadsPerBlock, d_M1, d_M2, d_Mout, n); 
    printf("Temps GPU (Multiplication): %f secondes\n", gpuTimeMult); 

    // Copie des résultats du GPU vers le CPU
    cudaMemcpy(Mout, d_Mout, size, cudaMemcpyDeviceToHost);

    // Optionnel : Affichage des résultats
    //MatrixPrint(Mout, n, p);

    // Libération mémoire
    free(M1);
    free(M2);
    free(Mout);
    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_Mout);

    return 0;
}