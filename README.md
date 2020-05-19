Tensorflow code for paper: Transductive Prototypical Network for Few-shot Classification (ICIP2020)

Requirements:
Python 3.6
Tensorflow 1.8.0
numpy
tqdm
opencv-python
pillow

Download data (miniImagenet and tieredImagenet):
Please download the compressed tar files from: https://github.com/renmengye/few-shot-ssl-public

Create a directory for miniImagenet:
mkdir -p data/mini-imagenet
mv mini-imagenet.tar.gz data/mini-imagenet
cd data/mini-imagenet
tar -zxvf mini-imagenet.tar.gz
rm -f mini-imagenet.tar.gz

Create a directory for tieredImagenet:
mkdir -p data/tiered-imagenet
mv tiered-imagenet.tar data/tiered-imagenet
cd data/tiered-imagenet
tar -xvf tiered-imagenet.tar
rm -f tiered-imagenet.tar

Train models for 5-way 1-shot setting:
miniImagenet: python Td_PN_train.py --gpu=0 --n_way=5 --n_shot=1 --n_test_way=5 --n_test_shot=1 --lr=0.001 --n_epochs=1000 --step_size=10000 --dataset=mini --exp_name=mini_Td_PN_5w1s_5tw1ts_alpha0.5_k10 --alpha=0.5 --k=10
tieredImagenet: python Td_PN_train.py --gpu=0 --n_way=5 --n_shot=1 --n_test_way=5 --n_test_shot=1 --lr=0.001 --n_epochs=3000 --step_size=40000 --dataset=tiered --exp_name=tiered_Td_PN_5w1s_5tw1ts_alpha0.5_k10 --alpha=0.5 --k=10
Other settings' trainings are similar

Test models for 5-way 1-shot setting:
miniImagenet: python Td_PN_test.py --gpu=0 --n_way=5 --n_shot=1 --n_test_way=5 --n_test_shot=1 --lr=0.001 --n_epochs=1000 --step_size=10000 --dataset=mini --exp_name=mini_Td_PN_5w1s_5tw1ts_alpha0.5_k10 --alpha=0.5 --k=10 --iters=80700
tieredImagenet: python Td_PN_train.py --gpu=0 --n_way=5 --n_shot=1 --n_test_way=5 --n_test_shot=1 --lr=0.001 --n_epochs=3000 --step_size=40000 --dataset=tiered --exp_name=tiered_Td_PN_5w1s_5tw1ts_alpha0.5_k10 --alpha=0.5 --k=10 --iters=298000
Other settings' testings are similar
