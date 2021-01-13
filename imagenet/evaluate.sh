python -u main.py \
--gpus 0 \
-e ./result/model_best.pth.tar \
--model resnet18_1w1a \
--data_path /media/ImageNet2012 \
--dataset imagenet \
-bt 128 \
--use_dali \