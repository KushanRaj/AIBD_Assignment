#include <bits/stdc++.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>


__device__ __forceinline__ float round_me(float var)
{
    // 37.66666 * 100 =3766.66
    // 3766.66 + .5 =3767.16    for rounding off value
    // then type cast to int so value is 3767
    // then divided by 100 so the value converted into 37.67
    float value = (int)(var * 10000 + .5);
    return (float)value / 10000;
}

__device__ __forceinline__ float atomicMaxFloat (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

__global__ void conv(float *weights, float *input, float *bias, float *output, int weightH, int H, int C, int outH, int outC){

    int img_id = blockIdx.x, feature_id = threadIdx.z, weight_id = blockIdx.z, pixel = blockIdx.y;
    float *my_filter = &weights[weight_id * weightH * weightH * C + feature_id * weightH * weightH];
    float *my_bias = &bias[weight_id];
    float *my_input = &input[img_id * C * H * H + feature_id * H * H];
    float *my_output = &output[img_id * outC * outH * outH + weight_id * outH * outH];

    int final_pid = (pixel % outH) * 2 + (2 * H * (pixel/outH));

    
    atomicAdd(&my_output[pixel], my_input[final_pid + H * threadIdx.y + threadIdx.x] * my_filter[threadIdx.x + weightH*threadIdx.y]);
    if (feature_id == 0 and threadIdx.x == 0 and threadIdx.y == 0){
      atomicAdd(&my_output[pixel], my_bias[0]);
    }

}


__global__ void conv_weight_backprop(float *weights, float *new_weights, float *new_bias,float *input, float *bias, float *output, int weightH, int H, int C, int outH, int outC, float learning_rate){


    int img_id = blockIdx.x, feature_id = threadIdx.z, weight_id = blockIdx.z, pixel = blockIdx.y;
    float *my_filter = &weights[weight_id * weightH * weightH * C + feature_id * weightH * weightH];
    float *my_new_filter = &new_weights[weight_id * weightH * weightH * C + feature_id * weightH * weightH];
    float my_bias = bias[weight_id];
    float *my_new_bias = &new_bias[weight_id];
    float *my_input = &input[img_id * C * H * H + feature_id * H * H];
    float *my_output = &output[img_id * outC * outH * outH + weight_id * outH * outH];

    int final_pid = (pixel % outH) * 2 + (2 * H * (pixel/outH));

    atomicAdd(&my_new_filter[threadIdx.x + weightH*threadIdx.y], my_filter[threadIdx.x + weightH*threadIdx.y]/(gridDim.x * gridDim.y) - (learning_rate * my_input[final_pid + H * threadIdx.y + threadIdx.x] * my_output[pixel]));

    if (threadIdx.x == 0 and threadIdx.y == 0 and feature_id == 0) {
      
      atomicAdd(&my_new_bias[0], my_bias/(gridDim.x * gridDim.y) - (learning_rate * my_output[pixel]));
    
      }
    
}

__global__ void conv_layer_backprop(float *weights, float *new_weights, float *new_bias,float *input, float *bias, float *output, int weightH, int H, int C, int outH, int outC, float learning_rate){


    int img_id = blockIdx.x, feature_id = threadIdx.z, weight_id = blockIdx.z, pixel = blockIdx.y;
    float *my_filter = &weights[weight_id * weightH * weightH * C + feature_id * weightH * weightH];
    float *my_input = &input[img_id * C * H * H + feature_id * H * H];
    float *my_output = &output[img_id * outC * outH * outH + weight_id * outH * outH];

    int final_pid = (pixel % outH) * 2 + (2 * H * (pixel/outH));

    atomicAdd(&my_input[final_pid + H * threadIdx.y + threadIdx.x], my_output[pixel] * my_filter[threadIdx.x + weightH*threadIdx.y]);

}

__global__ void relu(float *input, float *output, int H, int C){

    int img_id = blockIdx.x, feature_id = blockIdx.z, pixel = blockIdx.y;
    float *my_input = &input[img_id * C * H * H + feature_id * H * H];
    float *my_output = &output[img_id * C * H * H + feature_id * H * H];

    my_output[pixel] = max(my_input[pixel], 0.0);

}

__global__ void relu_backprop(float *input, float *output, int H, int C){

    int img_id = blockIdx.x, feature_id = blockIdx.z, pixel = blockIdx.y;
    float *my_input = &input[img_id * C * H * H + feature_id * H * H];
    float *my_output = &output[img_id * C * H * H + feature_id * H * H];

    if (my_input[pixel] < 0) my_input[pixel] = 0;
    else my_input[pixel] = my_output[pixel];

}


__global__ void maxpool(float *input, float *output, int window, int stride, int H, int C, int outH){

    int img_id = blockIdx.x, feature_id = blockIdx.z, pixel = blockIdx.y;

    float *my_input = &input[img_id * C * H * H + feature_id * H * H];

    float *my_output = &output[img_id * C * outH * outH + feature_id * outH * outH];

    int final_pid = (pixel % outH) * stride + (stride * H * (pixel/outH));

    __shared__ float w_max;

    if (threadIdx.x == 0 and threadIdx.y == 0) w_max = my_input[final_pid + H*threadIdx.y + threadIdx.x];

    __syncthreads();

    atomicMaxFloat(&w_max, my_input[final_pid + H*threadIdx.y + threadIdx.x]);

    __syncthreads();

    if (threadIdx.x == 0 and threadIdx.y == 0) {
      my_output[pixel] = w_max;

    }

}

__global__ void maxpool_backprop(float *input, float *output, int window, int stride, int H, int C, int outH){

    int img_id = blockIdx.x, feature_id = blockIdx.z, pixel = blockIdx.y;

    float *my_input = &input[img_id * C * H * H + feature_id * H * H];

    float *my_output = &output[img_id * C * outH * outH + feature_id * outH * outH];

    int final_pid = (pixel % outH) * stride + (stride * H * (pixel/outH));

    __shared__ float w_max;

    if (threadIdx.x == 0 and threadIdx.y == 0) w_max = my_input[final_pid + H*threadIdx.y + threadIdx.x];

    __syncthreads();

    atomicMaxFloat(&w_max, my_input[final_pid + H*threadIdx.y + threadIdx.x]);

    __syncthreads();

    if (w_max == my_input[final_pid + H*threadIdx.y + threadIdx.x]) {
      
      my_input[final_pid + H*threadIdx.y + threadIdx.x] = my_output[pixel];    
      }

    else my_input[final_pid + H*threadIdx.y + threadIdx.x] = 0;


}


__global__ void soft_backprop(float *output, int *target, int outdim, float loss_scale){

  __shared__ float base;

  if (threadIdx.x == 0) base = 0.0;

  __syncthreads();

  atomicAdd(&base, exp(output[blockIdx.x * outdim + threadIdx.x]));

  __syncthreads();

  if (threadIdx.x == target[blockIdx.x]) output[blockIdx.x * outdim + threadIdx.x] =  loss_scale * ((exp(output[blockIdx.x * outdim + threadIdx.x])/ base) - 1)/gridDim.x;
  else output[blockIdx.x * outdim + threadIdx.x] = loss_scale * (exp(output[blockIdx.x * outdim + threadIdx.x])/ base)/gridDim.x;


}

__global__ void get_loss(float *output, int *target, int *pred, int outdim, float *loss, int B){

  __shared__ float base;
  __shared__ volatile int w_max;
  __shared__ float v_max;

  if (threadIdx.x == 0) {

    w_max = threadIdx.x;
    v_max = output[blockIdx.x * outdim + threadIdx.x];
    base = 0.0;

  }


  __syncthreads();

  atomicAdd(&base, exp(output[blockIdx.x * outdim + threadIdx.x]));
  float old = atomicMaxFloat(&v_max, output[blockIdx.x * outdim + threadIdx.x]);

  __syncthreads();

  if (old == output[blockIdx.x * outdim + threadIdx.x]) w_max = threadIdx.x;

  __syncthreads();

  if (threadIdx.x == target[blockIdx.x]) {
    atomicAdd(loss, (-log(exp(output[blockIdx.x * outdim + threadIdx.x])/ base))/B); 
    pred[blockIdx.x] = w_max;
  }

}


struct pointers {
    float* weights;
    float* bias;
    float* output;
    float*  loss;
};

struct pointers CNN(int B, float *images, float *conv_weights, float *conv_bias, int *gt, int C, int hidC, int finC, int H, int WH, float learning_rate, float * loss, float* pred, float loss_scale){

  float *cuda_weights, *cuda_bias, *cuda_image, *outputs, *clr, *cl, *new_weights, *new_bias, *new_input, *clos;
  int *cgt, *c_pred;
  struct pointers output;

  cudaMalloc(&cuda_weights, ((hidC * C * WH * WH) + (finC * hidC * WH * WH) + (finC * finC * WH * WH)) * sizeof(float));
  cudaMalloc(&c_pred, B*sizeof(int));
  cudaMemset(c_pred, 0, B*sizeof(int));
  
  cudaMalloc(&new_weights, ((hidC * C * WH * WH) + (finC * hidC * WH * WH) + (finC * finC * WH * WH)) * sizeof(float));
  cudaMemset(new_weights, 0, ((hidC * C * WH * WH) + (finC * hidC * WH * WH) + (finC * finC * WH * WH)) * sizeof(float));
  cudaMemcpy(cuda_weights, conv_weights, ((hidC * C * WH * WH) + (finC * hidC * WH * WH) + (finC * finC * WH * WH)) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&cuda_bias, (hidC + finC + finC) * sizeof(float));
  cudaMalloc(&new_bias, (hidC + finC + finC) * sizeof(float));
  cudaMemset(new_bias, 0, (hidC + finC + finC) * sizeof(float));
  cudaMemcpy(cuda_bias, conv_bias, (hidC + finC + finC) * sizeof(float), cudaMemcpyHostToDevice);

  cudaHostAlloc(&clos, 1 * sizeof(float), 0);

  *clos = loss_scale;

  cudaMalloc(&cuda_image, (B * C * H * H) * sizeof(float));
  cudaMemcpy(cuda_image, images, (B * C * H * H) * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc(&clr, 1 * sizeof(float));
  cudaMemcpy(clr, &learning_rate, 1 * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc(&cgt, B * sizeof(int));
  cudaMemcpy(cgt, gt, B * sizeof(int), cudaMemcpyHostToDevice);

  cudaMalloc(&cl, 1 * sizeof(float));

  cudaMalloc(&outputs, ((B * hidC * (H/2) * (H/2)) + (B * hidC * (H/2) * (H/2)) + (B * hidC * (H/4) * (H/4)) + (B * finC * (H/8) * (H/8)) + (B * finC * (H/8) * (H/8)) + (B * finC * (H/16) * (H/16)) + (B * finC )) * sizeof(float));
  cudaMemset(outputs, 0, ((B * hidC * (H/2) * (H/2)) + (B * hidC * (H/2) * (H/2)) + (B * hidC * (H/4) * (H/4)) + (B * finC * (H/8) * (H/8)) + (B * finC * (H/8) * (H/8)) + (B * finC * (H/16) * (H/16)) + (B * finC )) * sizeof(float));
  cudaMalloc(&new_input, ((B * hidC * (H/2) * (H/2)) + (B * hidC * (H/2) * (H/2)) + (B * hidC * (H/4) * (H/4)) + (B * finC * (H/8) * (H/8)) + (B * finC * (H/8) * (H/8)) + (B * finC * (H/16) * (H/16)) + (B * finC )) * sizeof(float));
  cudaMemset(new_input, 0, ((B * hidC * (H/2) * (H/2)) + (B * hidC * (H/2) * (H/2)) + (B * hidC * (H/4) * (H/4)) + (B * finC * (H/8) * (H/8)) + (B * finC * (H/8) * (H/8)) + (B * finC * (H/16) * (H/16)) + (B * finC )) * sizeof(float));
  
  int outH = (H/2);

  cudaDeviceSynchronize();

  conv<<<dim3(B, outH * outH, hidC), dim3(WH, WH, C)>>>(cuda_weights, cuda_image, cuda_bias, outputs, WH, H, C, outH, hidC); 

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in conv1: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  relu<<<dim3(B, outH*outH, hidC), 1>>>(outputs, &outputs[B*hidC*(H/2)*(H/2)], outH, hidC) ; 

  err = cudaGetLastError();

  if (err != cudaSuccess) {
    printf("error in relu1: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  outH = outH/2;

  maxpool<<<dim3(B, outH*outH, hidC), dim3(2, 2, 1)>>>(&outputs[B*hidC*(H/2)*(H/2)], &outputs[ (B*hidC*(H/2)*(H/2)) * 2 ], 2, 2, outH * 2, hidC, outH) ; 

  err = cudaGetLastError();

  if (err != cudaSuccess) {
    printf("error in pool1: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  outH = outH/2;


  conv<<<dim3(B, outH * outH, finC), dim3(WH, WH, hidC)>>>(&cuda_weights[hidC * C * WH * WH], &outputs[ (B*hidC*(H/2)*(H/2)) * 2 ], &cuda_bias[hidC], &outputs[ (B*hidC*(H/2)*(H/2)) * 2 + (B * hidC * (H/4) * (H/4))], WH, outH * 2, hidC, outH, finC); 

  

  err = cudaGetLastError();

  

  if (err != cudaSuccess) {
    printf("error in conv2: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  relu<<<dim3(B, outH * outH, finC), 1>>>(&outputs[ (B*hidC*(H/2)*(H/2)) * 2 + (B * hidC * (H/4) * (H/4))], &outputs[(B * hidC * (H/2) * (H/2)) + (B * hidC * (H/2) * (H/2)) + (B * hidC * (H/4) * (H/4)) + (B * finC * (H/8) * (H/8))], outH, finC); 

  outH = outH/2;

  err = cudaGetLastError();

  if (err != cudaSuccess) {
    printf("error in relu2: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  maxpool<<<dim3(B, outH*outH, finC), dim3(2, 2, 1)>>>(&outputs[(B * hidC * (H/2) * (H/2)) + (B * hidC * (H/2) * (H/2)) + (B * hidC * (H/4) * (H/4)) + (B * finC * (H/8) * (H/8))], 
                                                     &outputs[(B * hidC * (H/2) * (H/2)) + (B * hidC * (H/2) * (H/2)) + (B * hidC * (H/4) * (H/4)) + (B * finC * (H/8) * (H/8)) + (B * finC * (H/8) * (H/8))], 2, 2, outH * 2, finC, outH) ;

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in pool2: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  outH = outH/2;

  conv<<<dim3(B, outH * outH, finC), dim3(WH, WH, finC)>>>(&cuda_weights[(hidC * C * WH * WH) + (finC * hidC * WH * WH)], &outputs[(B * hidC * (H/2) * (H/2)) + (B * hidC * (H/2) * (H/2)) + (B * hidC * (H/4) * (H/4)) + (B * finC * (H/8) * (H/8)) + (B * finC * (H/8) * (H/8))], 
                                                                      &cuda_bias[hidC + finC], &outputs[(B * hidC * (H/2) * (H/2)) + (B * hidC * (H/2) * (H/2)) + (B * hidC * (H/4) * (H/4)) + (B * finC * (H/8) * (H/8)) + (B * finC * (H/8) * (H/8)) + (B * finC * (H/16) * (H/16))], 
                                                                      WH, outH * 2, finC, outH, finC); 


  err = cudaGetLastError();

  if (err != cudaSuccess) {
    printf("error in conv3: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  get_loss<<<B, finC>>>(&outputs[(B * hidC * (H/2) * (H/2)) + (B * hidC * (H/2) * (H/2)) + (B * hidC * (H/4) * (H/4)) + (B * finC * (H/8) * (H/8)) + (B * finC * (H/8) * (H/8)) + (B * finC * (H/16) * (H/16))], cgt, c_pred, finC, cl, B);

  cudaMemcpy(pred, &outputs[(B * hidC * (H/2) * (H/2)) + (B * hidC * (H/2) * (H/2)) + (B * hidC * (H/4) * (H/4)) + (B * finC * (H/8) * (H/8)) + (B * finC * (H/8) * (H/8)) + (B * finC * (H/16) * (H/16))], B*finC*sizeof(int), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  if (learning_rate > 0){

    soft_backprop<<<B, finC>>>(&outputs[(B * hidC * (H/2) * (H/2)) + (B * hidC * (H/2) * (H/2)) + (B * hidC * (H/4) * (H/4)) + (B * finC * (H/8) * (H/8)) + (B * finC * (H/8) * (H/8)) + (B * finC * (H/16) * (H/16))], cgt, finC, *clos);
    
    err = cudaGetLastError();

    if (err != cudaSuccess) {
      printf("error in soft backprop: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }


    conv_weight_backprop<<<dim3(B, H/32 * H/32, finC), dim3(WH, WH, finC)>>>(&cuda_weights[(hidC * C * WH * WH) + (finC * hidC * WH * WH)], &new_weights[(hidC * C * WH * WH) + (finC * hidC * WH * WH)], &new_bias[(hidC) + (finC)], &outputs[(B * hidC * (H/2) * (H/2)) + (B * hidC * (H/2) * (H/2)) + (B * hidC * (H/4) * (H/4)) + (B * finC * (H/8) * (H/8)) + (B * finC * (H/8) * (H/8))], 
                                                                              &cuda_bias[(hidC ) + (finC )], &outputs[(B * hidC * (H/2) * (H/2)) + (B * hidC * (H/2) * (H/2)) + (B * hidC * (H/4) * (H/4)) + (B * finC * (H/8) * (H/8)) + (B * finC * (H/8) * (H/8)) + (B * finC * (H/16) * (H/16))], 
                                                                              WH, H/16, finC, H/32, finC, learning_rate);
    
    err = cudaGetLastError();

    if (err != cudaSuccess) {
      printf("error in conv backprop: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    conv_layer_backprop<<<dim3(B, H/32 * H/32, finC), dim3(WH, WH, finC)>>>(&cuda_weights[(hidC * C * WH * WH) + (finC * hidC * WH * WH)], &new_weights[(hidC * WH * WH) + (finC * hidC * WH * WH)], &new_bias[(hidC) + (finC)], &new_input[(B * hidC * (H/2) * (H/2)) + (B * hidC * (H/2) * (H/2)) + (B * hidC * (H/4) * (H/4)) + (B * finC * (H/8) * (H/8)) + (B * finC * (H/8) * (H/8))], 
                                                                              &cuda_bias[(hidC) + (finC)], &outputs[(B * hidC * (H/2) * (H/2)) + (B * hidC * (H/2) * (H/2)) + (B * hidC * (H/4) * (H/4)) + (B * finC * (H/8) * (H/8)) + (B * finC * (H/8) * (H/8)) + (B * finC * (H/16) * (H/16))], 
                                                                              WH, H/16, finC, H/32, finC, learning_rate);


    err = cudaGetLastError();

    if (err != cudaSuccess) {
      printf("error in conv backprop: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    maxpool_backprop<<<dim3(B, H/16*H/16, finC), dim3(2, 2, 1)>>>(&outputs[(B * hidC * (H/2) * (H/2)) + (B * hidC * (H/2) * (H/2)) + (B * hidC * (H/4) * (H/4)) + (B * finC * (H/8) * (H/8))], 
                                                      &new_input[(B * hidC * (H/2) * (H/2)) + (B * hidC * (H/2) * (H/2)) + (B * hidC * (H/4) * (H/4)) + (B * finC * (H/8) * (H/8)) + (B * finC * (H/8) * (H/8))], 2, 2, H/8, finC, H/16);
    

    err = cudaGetLastError();

    if (err != cudaSuccess) {
      printf("error in conv3 backprop: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    relu_backprop<<<dim3(B, H/8*H/8, finC), 1>>>(&outputs[ (B*hidC*(H/2)*(H/2)) * 2 + (B * hidC * (H/4) * (H/4))], &outputs[(B * hidC * (H/2) * (H/2)) + (B * hidC * (H/2) * (H/2)) + (B * hidC * (H/4) * (H/4)) + (B * finC * (H/8) * (H/8))], H/8, finC); 


    err = cudaGetLastError();

    if (err != cudaSuccess) {
      printf("error in conv3 backprop: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    conv_weight_backprop<<<dim3(B, H/8 * H/8, finC), dim3(WH, WH, hidC)>>>(&cuda_weights[hidC * C * WH * WH], &new_weights[hidC * C * WH * WH], &new_bias[hidC], &outputs[ (B*hidC*(H/2)*(H/2)) * 2 ], 
                                                                              &cuda_bias[hidC], &outputs[ (B*hidC*(H/2)*(H/2)) * 2 + (B * hidC * (H/4) * (H/4))], 
                                                                              WH, H/4, hidC, H/8, finC, learning_rate);
    
    err = cudaGetLastError();

    if (err != cudaSuccess) {
      printf("error in conv backprop: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    conv_layer_backprop<<<dim3(B, H/8 * H/8, finC), dim3(WH, WH, hidC)>>>(&cuda_weights[hidC * C * WH * WH], &new_weights[hidC * C * WH * WH], &new_bias[hidC], &new_input[ (B*hidC*(H/2)*(H/2)) * 2 ], 
                                                                              &cuda_bias[hidC], &outputs[ (B*hidC*(H/2)*(H/2)) * 2 + (B * hidC * (H/4) * (H/4))], 
                                                                              WH, H/4, hidC, H/8, finC, learning_rate);
                                                                        
    
    err = cudaGetLastError();

    if (err != cudaSuccess) {
      printf("error in conv backprop: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    maxpool_backprop<<<dim3(B, H/4*H/4, hidC), dim3(2, 2, 1)>>>(&outputs[B*hidC*(H/2)*(H/2)], 
                                                      &new_input[ (B*hidC*(H/2)*(H/2)) * 2 ], 2, 2, H/2, hidC, H/4);
    
    err = cudaGetLastError();

    if (err != cudaSuccess) {
      printf("error in conv3 backprop: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    relu_backprop<<<dim3(B, H/2*H/2, hidC), 1>>>(outputs, &outputs[B*hidC*(H/2)*(H/2)], H/2, hidC);

    err = cudaGetLastError();

    if (err != cudaSuccess) {
      printf("error in conv3 backprop: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    conv_weight_backprop<<<dim3(B, H/2 * H/2, hidC), dim3(WH, WH, C)>>>(cuda_weights, new_weights, new_bias, cuda_image, 
                                                                              cuda_bias, outputs, 
                                                                              WH, H, C, H/2, hidC, learning_rate);

    err = cudaGetLastError();

    if (err != cudaSuccess) {
      printf("error in conv backprop: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }

  cudaMemcpy(loss, cl, 1 * sizeof(int), cudaMemcpyDeviceToHost);  
  cudaMemcpy(conv_weights, new_weights, ((hidC * C * WH * WH) + (finC * hidC * WH * WH) + (finC * finC * WH * WH)) * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(conv_bias, new_bias, ((hidC ) + (finC ) + (finC)) * sizeof(float), cudaMemcpyDeviceToHost);
  

  cudaFree(new_weights);
  cudaFree(new_bias);
  cudaFree(c_pred);
  //cudaFree(cl);
  cudaFree(new_input);
  cudaFree(outputs);
  cudaFree(cuda_image);
  cudaFree(cuda_bias);
  cudaFree(cuda_weights);
  //cudaFree(cgt);
  //cudaFree(c_pred);
  cudaFree(clr);
  //cudaFree(clos);

  cudaDeviceSynchronize();


  output.weights = conv_weights;
  output.bias = conv_bias;
  output.output = pred;
  output.loss = loss;

  return output;
}
