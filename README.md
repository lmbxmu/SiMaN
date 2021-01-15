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