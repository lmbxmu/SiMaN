python -u main.py \
--gpus 0 \
-e [model_best.pth.tar] \
--model resnet20_1w1a \
--data_path /data \
--dataset cifar10 \
-bt 128 \