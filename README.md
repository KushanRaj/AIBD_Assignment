# AIBD_Assignment
The repository contains a fully custom coded CUDA kernel to perform 2D convolutions for the task for classification of images, in out case the CIFAR dataset. In addition the codebase also contains a model utilising 3D convolutions.

## Important!
When using either 2D or 3D method, please remember to `cd` into the specific folder.

In addition if using 2D convolution with the CUDA kernel plaese remember to complie it via
```
cd CUDAConv
python3 setup.py build develop --user
```
Please also ensure to download the required packages mentioned in the `requirements.txt` file.

In the 2D case, we also provide a `test` model which is essentially the CUDA coded model written using pytorch to compare and evaluate our model.

## Training the model

Run the train.py file in either folder to train the 2D/3D model. Provide the location of the downloaded data using `--data_root`. If the deata has not been downloaded, torch will download it on its own.

In addition, the parameters can be tuned by changing the `--hid_dims` argument. In addition, `--aug` can be provided as a parameter to add augmentaions to potentially make the model more robust. To see more options, please refer to the `utils.py` to see all available parameters.

## Plotting

`--wandb` can be passed as aparamter to use WandB as tracker.
