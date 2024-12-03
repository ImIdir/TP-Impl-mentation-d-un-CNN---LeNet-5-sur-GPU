#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void MatrixPrint(float *M, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            printf("%f ", M[i * p + j]);
        }
        printf("\n");
    }
}

float random_float(float min, float max) {
    return min + (max - min) * rand() / (float)RAND_MAX;
}

void init_input_data(float* raw_data) {
    for(int i = 0; i < 32*32; i++) {
        raw_data[i] = random_float(0, 1);
    }
}

void init_conv_kernels(float* C1_kernel) {
    for(int k = 0; k < 6; k++) {
        for(int i = 0; i < 5*5; i++) {
            C1_kernel[k*25 + i] = random_float(0, 1);
        }
    }
}

__device__ float activation_tanh(float x) {
    return tanh(x);
}

// Kernel pour la convolution 2D :
__global__ void conv2D(float* input, float* kernels, float* output, int input_size, int kernel_size) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    int feature_map = blockIdx.z;
    
    if(tx < 28 && ty < 28 && feature_map < 6) {
        float sum = 0.0f;
        
        for(int i = 0; i < kernel_size; i++) {
            for(int j = 0; j < kernel_size; j++) {
                int x = tx + i;
                int y = ty + j;
                sum += input[y * input_size + x] * 
                       kernels[feature_map * kernel_size * kernel_size + i * kernel_size + j];
            }
        }
        
        output[feature_map * 28 * 28 + ty * 28 + tx] = activation_tanh(sum);
    }
}

// Kernel pour le sous-échantillonnage (Layer 3) :
__global__ void subsample(float* input, float* output) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    int feature_map = blockIdx.z;
    
    if(tx < 14 && ty < 14 && feature_map < 6) {
        float sum = 0.0f;
        
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 2; j++) {
                sum += input[feature_map * 28 * 28 + (2*ty + i) * 28 + (2*tx + j)];
            }
        }
        
        output[feature_map * 14 * 14 + ty * 14 + tx] = sum / 4.0f;
    }
}

int main() {

    srand(time(NULL));
    
    float *raw_data = (float*)malloc(32 * 32 * sizeof(float));
    float *C1_kernel = (float*)malloc(6 * 5 * 5 * sizeof(float));
    float *C1_data = (float*)malloc(6 * 28 * 28 * sizeof(float));
    float *S1_data = (float*)malloc(6 * 14 * 14 * sizeof(float));
    
    // Initialisation des données
    init_input_data(raw_data);
    init_conv_kernels(C1_kernel);

       // Affichage des données d'entrée
    printf("\n=== COUCHE 1 - ENTRÉE (32x32) ===\n");
    MatrixPrint(raw_data, 32, 32);
    
    // Affichage des kernels de convolution
    printf("\n=== KERNELS DE CONVOLUTION (6 kernels de 5x5) ===\n");
    for(int k = 0; k < 6; k++) {
        printf("\nKernel %d:\n", k+1);
        MatrixPrint(&C1_kernel[k*25], 5, 5);
    }
    
    float *d_raw_data, *d_C1_kernel, *d_C1_data, *d_S1_data;
    cudaMalloc(&d_raw_data, 32 * 32 * sizeof(float));
    cudaMalloc(&d_C1_kernel, 6 * 5 * 5 * sizeof(float));
    cudaMalloc(&d_C1_data, 6 * 28 * 28 * sizeof(float));
    cudaMalloc(&d_S1_data, 6 * 14 * 14 * sizeof(float));
    
    cudaMemcpy(d_raw_data, raw_data, 32 * 32 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, 6 * 5 * 5 * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 blockDim_conv(16, 16);
    dim3 gridDim_conv((28 + blockDim_conv.x - 1) / blockDim_conv.x,
                      (28 + blockDim_conv.y - 1) / blockDim_conv.y,
                      6);
    
    // convolution
    conv2D<<<gridDim_conv, blockDim_conv>>>(d_raw_data, d_C1_kernel, d_C1_data, 32, 5);

    // Copie des résultats vers le CPU
    cudaMemcpy(C1_data, d_C1_data, 6 * 28 * 28 * sizeof(float), cudaMemcpyDeviceToHost);

    // affichage des résultats de convolution
    printf("\n=== COUCHE 2 - APRÈS CONVOLUTION (6 feature maps de 28x28) ===\n");
    for(int k = 0; k < 6; k++) {
        printf("\nFeature Map %d:\n", k+1);
        MatrixPrint(&C1_data[k*28*28], 28, 28);
    }
    
    dim3 blockDim_sub(14, 14);
    dim3 gridDim_sub(1, 1, 6);
    
    // sous-échantillonnage
    subsample<<<gridDim_sub, blockDim_sub>>>(d_C1_data, d_S1_data);
    
    // Copie des résultats vers le CPU
    cudaMemcpy(S1_data, d_S1_data, 6 * 14 * 14 * sizeof(float), cudaMemcpyDeviceToHost);

    //affichage des résultats de sous-échantillonnage
    printf("\n=== COUCHE 3 - APRÈS SOUS-ÉCHANTILLONNAGE (6 feature maps de 14x14) ===\n");
    for(int k = 0; k < 6; k++) {
        printf("\nFeature Map %d:\n", k+1);
        MatrixPrint(&S1_data[k*14*14], 14, 14);
    }
    
    free(raw_data);
    free(C1_kernel);
    free(C1_data);
    free(S1_data);
    cudaFree(d_raw_data);
    cudaFree(d_C1_kernel);
    cudaFree(d_C1_data);
    cudaFree(d_S1_data);
    
    return 0;
}