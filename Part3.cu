#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

// Structure pour les poids de LeNet-5
typedef struct {
    // Conv1: 6 kernels 5x5
    float *conv1_weights;  // [6][5][5]
    float *conv1_bias;     // [6]
    
    // Conv2: 16 kernels 5x5x6
    float *conv2_weights;  // [16][6][5][5]
    float *conv2_bias;     // [16]
    
    // Dense1: 120 neurones
    float *fc1_weights;    // [120][400]
    float *fc1_bias;      // [120]
    
    // Dense2: 84 neurones
    float *fc2_weights;    // [84][120]
    float *fc2_bias;      // [84]
    
    // Output: 10 classes
    float *fc3_weights;    // [10][84]
    float *fc3_bias;      // [10]
} LeNet5Weights;

// Structure pour les données intermédiaires sur GPU
typedef struct {
    float *d_input;         // [28][28]
    float *d_conv1_output;  // [6][24][24]
    float *d_pool1_output;  // [6][12][12]
    float *d_conv2_output;  // [16][8][8]
    float *d_pool2_output;  // [16][4][4]
    float *d_flatten;       // [400]
    float *d_fc1_output;    // [120]
    float *d_fc2_output;    // [84]
    float *d_fc3_output;    // [10]
    float *d_final_output;  // [10]
} LeNet5Buffers;

// Fonction de chargement des poids depuis le fichier .h5
void loadWeightsFromH5(const char* filename, LeNet5Weights* weights) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Erreur: Impossible d'ouvrir le fichier de poids\n");
        return;
    }
    
    // Lecture des poids dans l'ordre
    fread(weights->conv1_weights, sizeof(float), 6*5*5, file);
    fread(weights->conv1_bias, sizeof(float), 6, file);
    fread(weights->conv2_weights, sizeof(float), 16*6*5*5, file);
    fread(weights->conv2_bias, sizeof(float), 16, file);
    fread(weights->fc1_weights, sizeof(float), 120*400, file);
    fread(weights->fc1_bias, sizeof(float), 120, file);
    fread(weights->fc2_weights, sizeof(float), 84*120, file);
    fread(weights->fc2_bias, sizeof(float), 84, file);
    fread(weights->fc3_weights, sizeof(float), 10*84, file);
    fread(weights->fc3_bias, sizeof(float), 10, file);
    
    fclose(file);
}

// Première couche de convolution
__global__ void conv2D_layer1(float* input, float* kernels, float* bias, float* output) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    int feature_map = blockIdx.z;
    
    if(tx < 24 && ty < 24 && feature_map < 6) {
        float sum = 0.0f;
        
        // Convolution 5x5
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < 5; j++) {
                sum += input[(ty+i)*28 + (tx+j)] * kernels[feature_map*25 + i*5 + j];
            }
        }
        
        sum += bias[feature_map];
        output[feature_map*576 + ty*24 + tx] = tanhf(sum);
    }
}

// Premier average pooling
__global__ void avgpool2D_layer1(float* input, float* output) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    int feature_map = blockIdx.z;
    
    if(tx < 12 && ty < 12 && feature_map < 6) {
        float sum = 0.0f;
        
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 2; j++) {
                sum += input[feature_map*576 + (2*ty+i)*24 + (2*tx+j)];
            }
        }
        
        output[feature_map*144 + ty*12 + tx] = sum / 4.0f;
    }
}

// Deuxième convolution
__global__ void conv2D_layer2(float* input, float* kernels, float* bias, float* output) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    int feature_map = blockIdx.z;
    
    if(tx < 8 && ty < 8 && feature_map < 16) {
        float sum = 0.0f;
        
        for(int input_channel = 0; input_channel < 6; input_channel++) {
            for(int i = 0; i < 5; i++) {
                for(int j = 0; j < 5; j++) {
                    float input_val = input[input_channel*144 + (ty+i)*12 + (tx+j)];
                    float kernel_val = kernels[feature_map*150 + input_channel*25 + i*5 + j];
                    sum += input_val * kernel_val;
                }
            }
        }
        
        sum += bias[feature_map];
        output[feature_map*64 + ty*8 + tx] = tanhf(sum);
    }
}

// Deuxième average pooling
__global__ void avgpool2D_layer2(float* input, float* output) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    int feature_map = blockIdx.z;
    
    if(tx < 4 && ty < 4 && feature_map < 16) {
        float sum = 0.0f;
        
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 2; j++) {
                sum += input[feature_map*64 + (2*ty+i)*8 + (2*tx+j)];
            }
        }
        
        output[feature_map*16 + ty*4 + tx] = sum / 4.0f;
    }
}

// Flatten
__global__ void flatten(float* input, float* output) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(idx < 400) {
        int fm = idx / 16;
        int pos = idx % 16;
        int y = pos / 4;
        int x = pos % 4;
        output[idx] = input[fm*16 + y*4 + x];
    }
}

// Dense layers
__global__ void dense_layer(float* input, float* weights, float* bias, 
                          float* output, int input_size, int output_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(idx < output_size) {
        float sum = 0.0f;
        for(int i = 0; i < input_size; i++) {
            sum += input[i] * weights[idx*input_size + i];
        }
        sum += bias[idx];
        output[idx] = tanhf(sum);
    }
}

// Softmax output layer
__global__ void softmax_layer(float* input, float* weights, float* bias, float* output) {
    __shared__ float max_val;
    __shared__ float sum;
    
    if(threadIdx.x == 0) {
        max_val = -INFINITY;
        for(int i = 0; i < 10; i++) {
            float val = 0.0f;
            for(int j = 0; j < 84; j++) {
                val += input[j] * weights[i*84 + j];
            }
            val += bias[i];
            max_val = fmaxf(max_val, val);
        }
    }
    __syncthreads();
    
    int idx = threadIdx.x;
    if(idx < 10) {
        float val = 0.0f;
        for(int i = 0; i < 84; i++) {
            val += input[i] * weights[idx*84 + i];
        }
        val += bias[idx];
        output[idx] = expf(val - max_val);
    }
    __syncthreads();
    
    if(threadIdx.x == 0) {
        sum = 0.0f;
        for(int i = 0; i < 10; i++) {
            sum += output[i];
        }
    }
    __syncthreads();
    
    if(idx < 10) {
        output[idx] /= sum;
    }
}

// Fonction principale d'inférence
void lenet5_inference(float* input_image, LeNet5Weights* weights, LeNet5Buffers* buffers) {
    // Paramètres des kernels
    dim3 block_conv1(8, 8);
    dim3 grid_conv1(3, 3, 6);
    
    dim3 block_pool1(12, 12);
    dim3 grid_pool1(1, 1, 6);
    
    dim3 block_conv2(8, 8);
    dim3 grid_conv2(1, 1, 16);
    
    dim3 block_pool2(4, 4);
    dim3 grid_pool2(1, 1, 16);
    
    // Copie de l'image d'entrée vers le GPU
    cudaMemcpy(buffers->d_input, input_image, 28*28*sizeof(float), cudaMemcpyHostToDevice);
    
    // Conv1
    conv2D_layer1<<<grid_conv1, block_conv1>>>(
        buffers->d_input, weights->conv1_weights, weights->conv1_bias, buffers->d_conv1_output);
    
    // Pool1
    avgpool2D_layer1<<<grid_pool1, block_pool1>>>(
        buffers->d_conv1_output, buffers->d_pool1_output);
    
    // Conv2
    conv2D_layer2<<<grid_conv2, block_conv2>>>(
        buffers->d_pool1_output, weights->conv2_weights, weights->conv2_bias, buffers->d_conv2_output);
    
    // Pool2
    avgpool2D_layer2<<<grid_pool2, block_pool2>>>(
        buffers->d_conv2_output, buffers->d_pool2_output);
    
    // Flatten
    flatten<<<(400+255)/256, 256>>>(
        buffers->d_pool2_output, buffers->d_flatten);
    
    // Dense1
    dense_layer<<<(120+255)/256, 256>>>(
        buffers->d_flatten, weights->fc1_weights, weights->fc1_bias, 
        buffers->d_fc1_output, 400, 120);
    
    // Dense2
    dense_layer<<<(84+255)/256, 256>>>(
        buffers->d_fc1_output, weights->fc2_weights, weights->fc2_bias,
        buffers->d_fc2_output, 120, 84);
    
    // Output avec softmax
    softmax_layer<<<1, 10>>>(
        buffers->d_fc2_output, weights->fc3_weights, weights->fc3_bias,
        buffers->d_final_output);
}

int main() {
    // Allocation des poids sur CPU
    LeNet5Weights weights;
    // Allouer la mémoire pour les poids...
    
    // Charger les poids
    loadWeightsFromH5("lenet5_weights.h5", &weights);
    
    // Allocation des buffers sur GPU
    LeNet5Buffers buffers;
    cudaMalloc(&buffers.d_input, 28*28*sizeof(float));
    cudaMalloc(&buffers.d_conv1_output, 6*24*24*sizeof(float));
    cudaMalloc(&buffers.d_pool1_output, 6*12*12*sizeof(float));
    cudaMalloc(&buffers.d_conv2_output, 16*8*8*sizeof(float));
    cudaMalloc(&buffers.d_pool2_output, 16*4*4*sizeof(float));
    cudaMalloc(&buffers.d_flatten, 400*sizeof(float));
    cudaMalloc(&buffers.d_fc1_output, 120*sizeof(float));
    cudaMalloc(&buffers.d_fc2_output, 84*sizeof(float));
    cudaMalloc(&buffers.d_fc3_output, 10*sizeof(float));
    cudaMalloc(&buffers.d_final_output, 10*sizeof(float));
    
    // Copier les poids sur le GPU...
    
    // Exemple d'utilisation
    float input_image[28*28];
    // Charger une image...
    
    // Faire l'inférence
    lenet5_inference(input_image, &weights, &buffers);
    
    // Récupérer le résultat
    float output[10];
    cudaMemcpy(output, buffers.d_final_output, 10*sizeof(float), cudaMemcpyDeviceToHost);
    
    // Libérer la mémoire
    // ...
    
    return 0;
}