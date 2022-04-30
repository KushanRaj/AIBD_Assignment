#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

struct pointers {
    float* weights;
    float* bias;
    float* output;
    float*  loss;
};

struct pointers CNN(int B, float *images, float *conv_weights, float *conv_bias, int *gt, int C, int hidC, int finC, int H, int WH, float learning_rate, float *loss, float *pred, float loss_scale);

struct pointers conv2D(int B, float *images, float *conv_weights, float *conv_bias, int *gt, int C, int hidC, int finC, int H, int WH, float learning_rate, float *loss, float *pred, float loss_scale){

  return CNN(B, images, conv_weights, conv_bias, gt, C, hidC, finC, H, WH, learning_rate, loss, pred, loss_scale);

}

std::vector<torch::Tensor> myCNN(
  int B,
  torch::Tensor t_images,
  torch::Tensor t_conv_weights,
  torch::Tensor t_conv_bias,
  torch::Tensor t_gt, 
  torch::Tensor t_loss,
  torch::Tensor t_pred,
  int C, 
  int hidC, int finC, int H, int WH, float learning_rate, float loss_scale
  ) {

  struct pointers out;

  float *conv_weights, *conv_bias, *images, *loss, *pred;
  int *gt;

  conv_weights = t_conv_weights.data_ptr<float>();
  conv_bias = t_conv_bias.data_ptr<float>();
  images = t_images.data_ptr<float>();
  gt = t_gt.data_ptr<int>();
  pred = t_pred.data_ptr<float>();
  loss = t_loss.data_ptr<float>();

  out = conv2D(B, images, conv_weights, conv_bias, gt, C, hidC, finC, H, WH, learning_rate, loss, pred, loss_scale);
    
  return {t_conv_weights, t_conv_bias, t_loss, t_pred};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("CNN", &myCNN, "Conv2D forward (CUDA)");
}

