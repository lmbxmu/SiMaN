python -u main.py \
--gpus 0 \
-e /home/sdb/xuzihan/ICML2021/SOTAs/vgg/vgg_prelu_initPercent5_seed1234_400epoch_92.53/model_best.pth.tar \
--model vgg_small_1w1a \
--data_path /home/xuzihan/data \
--dataset cifar10 \
-bt 128 \