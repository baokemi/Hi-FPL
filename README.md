
#  Project Guide

Welcome to the Hi-FPL project's training guide. 
Hi-FPL is a novel Hierarchical Federated Prompt Learning framework that aims to construct structurally-aware and rapidly adaptive prompt tokens for both generalization and personalization in FL.


## 1. command

```
python main.py  --pretrained_dir checkpoints/imagenet21k_ViT-B_16.npz --model_type ViT-B_16 --partition noniid-labeluni --n_parties 100  --cls_num 10 --device cuda:0 --batch_size 128 --comm_round 150  --test_round 0 --sample 0.05  --rho 0.9 --alg HiFPL --dataset cifar100 --client_lr 0.1 --corr_lr 0.00001  --dac_lr 0.000001 --client_epochs 10  --server_epochs 10 --share_blocks 0 1 2 3 4 --share_blocks_g 5 --prompt_sample_type max_pooling --n_prompt 4 --client_gam 9 --init_seed 0
```
