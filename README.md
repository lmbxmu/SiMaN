# ICML21

## Training on CIFAR-10
```bash
python -u main.py \
--gpus 0 \
--seed 1234 \
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
|[resnet18_1w1a](https://drive.google.com/drive/folders/1aZ48yGxp6KTmGUw4yhGW2U6gdRBRwRh6?usp=sharing) |    256     |       128       | 400   | 92.5 |  
|[resnet20_1w1a](https://drive.google.com/drive/folders/1UP9fxm_60LmgR87BKU3S9DkziAT77NZg?usp=sharing) |    256     |       128       | 400   | 87.4 |
|[vgg_small_1w1a](https://drive.google.com/drive/folders/1zHwJCX3Hn-EeZcDRz0et6NByuv5n7Tno?usp=sharing) |    256     |       128       | 400   | 92.5 |

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
--weight_decay 1e-4 \
--use_dali \
```

### Results on ImageNet

|Quantized model Link                                                                                  | batch_size | batch_size_test | epochs| use_dali| Top-1 | Top-5 | 
|:----------------------------------------------------------------------------------------------------:|:----------:|:---------------:|:-----:|:-------:|:-----:|:-----:|
| [resnet18_1w1a](https://drive.google.com/drive/folders/1xujH6ko6GMtg32hvXvcHB8_VRdjwf7JV?usp=sharing)|    512     |       256       |  150  |   ✔   | 60.1 | 82.3 |
| [resnet34_1w1a](https://drive.google.com/drive/folders/1VadtN4wDjBMRuOakzvH6n5g4MSZAxDYV?usp=sharing)|    512     |       256       |  150  |   ✔   | 63.1 | 84.3 |