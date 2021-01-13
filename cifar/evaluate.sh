python -u main.py \
--gpus 0 \
-e /home/sdb/xuzihan/ICML2021/SOTAs/res18/resnet18_prelu_initPercent5_seed123_400epoch_92.49/model_best.pth.tar \
--model resnet18_1w1a \
--data_path /home/xuzihan/data \
--dataset cifar10 \
-bt 128 \