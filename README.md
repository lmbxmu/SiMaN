# ICML21
Pytorch implementation of ICML2021_SiMaN   

## Dependencies 
* python 3.7
* pytorch 1.1.0 
* torchvision 0.3.0
* numpy 1.17.2

## Training on CIFAR-10
```bash
python -u main.py \
--gpus 0 \
--seed 123 \
--model resnet18_1w1a (or resnet20_1w1a or vgg_small_1w1a) \
--results_dir ./result \
--data_path [DATA_PATH] \
--dataset cifar10 \
--epochs 400 \
--lr 0.1 \
-b 256 \
-bt 128 \
--lr_type cos \
--warm_up \
--weight_decay 5e-4 \
```

### Results on CIFAR-10. 
|Quantized model Link                                                                                  | batch_size | batch_size_test | epochs| Top-1 |
|:----------------------------------------------------------------------------------------------------:|:----------:|:---------------:|:-----:|:-----:|
|[resnet18_1w1a](https://drive.google.com/drive/folders/1x2ihCroNVhW08drZJUTyI3XIEQpDc2su?usp=sharing) |    256     |       128       | 400   | 92.5 |  
|[resnet20_1w1a](https://drive.google.com/drive/folders/1-1HXFFsw0bGplsA-pkYetl-ETFpvLGiL?usp=sharing) |    256     |       128       | 400   | 87.4 |
|[vgg_small_1w1a](https://drive.google.com/drive/folders/1vkV_U63kXxBumYT-OVV80mesKpAzFztJ?usp=sharing) |    256     |       128       | 400   | 92.5 |


To ensure the reproducibility, please refer to our training details provided in the links for our quantized models.

To verify the performance of our quantized models on CIFAR-10, please use the following command:
```bash 
python -u main.py \
--gpus 0 \
-e [best_model_path] \
--model resnet20_1w1a (resnet18_1w1a or vgg_small_1w1a) \
--data_path [DATA_PATH] \
--dataset cifar10 \
-bt 128 \
```

## Training on ImageNet
```bash
python -u main.py \
--gpus 0,1 \
--model resnet18_1w1a (or resnet34_1w1a) \
--results_dir ./result \
--data_path [DATA_PATH] \
--dataset imagenet \
--epochs 150 \
--lr 0.1 \
-b 512 \
-bt 256 \
--lr_type cos \
--use_dali \
```

We provide two types of dataloaders by [nvidia-dali](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html) and [Pytorch](https://pytorch.org/docs/stable/data.html) respectively. They use the same data augmentations, including random crop and horizontal flip. We empirically find that the dataloader by Pytorch can offer a better accuracy performance. They may have different code implementations. Anyway, we haven't figured it out yet. However, nvidia-dali shows its extreme efficiency in processing data which well accelerates the network training. The reported experimental results are on the basis of nvidia-dali. If interested, you can try dataloader by Pytorch via removing the optional argument ```--use_dali``` to obtain a better performance.  

Nvidia-dali package
```bash
# for CUDA 10
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100
# for CUDA 11
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110
```
### Results on ImageNet

|Quantized model Link                                                                                  | batch_size | batch_size_test | epochs| use_dali| Top-1 | Top-5 | 
|:----------------------------------------------------------------------------------------------------:|:----------:|:---------------:|:-----:|:-------:|:-----:|:-----:|
| [resnet18_1w1a](https://drive.google.com/drive/folders/15pwL5UeJGFHNwHNFh7Yl6dFgAX9QvxE5?usp=sharing)|    512     |       256       |  150  |   ✔    | 60.1 | 82.3 |
| [resnet34_1w1a](https://drive.google.com/drive/folders/1vhl1Q9ulTfqMy27Gn5lFIMgop24UIRh8?usp=sharing)|    512     |       256       |  150  |   ✔    | 63.9 | 84.8 |

To ensure the reproducibility, please refer to our training details provided in the links for our quantized models. \
Small tips for further boosting the performance of our method: (1) removing the optional argument ```--use_dali``` as discussed above; (2) increasing the training epochs; (3) enlarging the batch size for training.

To verify the performance of our quantized models on ImageNet, please use the following command:
```bash
python -u main.py \
--gpu 0 \
-e [best_model_path] \
--model resnet18_1w1a (or resnet34_1w1a)\
--dataset imagenet \
--data_path [DATA_PATH] \
-bt 256 \
--use dali
```