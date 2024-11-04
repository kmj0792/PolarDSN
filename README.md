# PolarDSN: An Inductive Approach to Learning the Evolution of Network Polarization in Dynamic Signed Networks

![git_overview](https://github.com/user-attachments/assets/5962613d-ddca-4f50-92ed-cbb8b69e2d89)

This repository provides a reference implementation of PolarDSN as described in the following paper "[PolarDSN: An Inductive Approach to Learning the Evolution of Network Polarization in Dynamic Signed Networks](https://doi.org/10.1145/3627673.3679654)", published at CIKM 2024 (full paper). (33rd ACM International Conference on Information and Knowledge Management (ACM CIKM 2024))

## Authors
- Min-Jeong Kim (kmj0792@hanyang.ac.kr)
- Yeon-Chang Lee (yeonchang@unist.ac.kr)
- Sang-Wook Kim (wook@hanyang.ac.kr)

## Inputs
The input dataset should be saved in ```./DynamicData/weight/``` folder. 


<img src = "https://github.com/user-attachments/assets/88e73895-37e3-4bd2-85e7-e95fe095f955" width="500" height="200">

The structure of the input dataset is the following: ```| node_id1 | node_id2 timestamp | label(=sign) | weight | idx |```

Node ids start from 1 to |*V*| (*V* is the set of nodes in the dataset).

## Outputs
The accuracies of PolarDSN are saved in ```./PolarDSN/log``` folder. 

## Arguments
#### Select dataset and training mode 
```
--data                    Dataset name. (default: "bitcoinalpha")
```

#### Method-related hyper-parameters
```
--n_degree                The number of paths(i.e., alpha). (default: 32)
--n_layer                 The length of each path(i.e., beta). (default: 3)
--bias                    Temporal decay (i.e., gamma). (default: 1e-6)
```

## Usage
To run SVD-AE on different datasets, use the following commands:

+ For Bitcoin-alpha:
```
python main.py --data=bitcoinalpha --bs=64 --lr=0.001 --n_degree=64 --n_layer=3 --train_time_encoding=learn --edge_embedding=concat --neigh_agg=rnn --path_agg=mean --bias=1e-6 --seed=0 --walk_type=before --direct=add --co_occ=learn_add --gpu=0
```

+ For Bitcoin-otc:
```
python main.py --data=bitcoinotc --bs=64 --lr=0.001 --n_degree=128 --n_layer=5 --train_time_encoding=nonlearn --edge_embedding=concat --neigh_agg=rnn --path_agg=mean --bias=1e-6 --seed=0 --walk_type=before --direct=add --co_occ=learn_add --gpu=0
```

+ For Wiki-RfA:
```
python main.py --data=wiki-RfA --bs=64 --lr=0.001 --n_degree=64 --n_layer=2 --train_time_encoding=nonlearn --edge_embedding=concat --neigh_agg=rnn --path_agg=mean --bias=1e-7 --seed=0 --walk_type=before --direct=add --co_occ=learn_add --gpu=0
```

+ For Epinions:
```
python main.py --data=epinions --bs=64 --lr=0.001 --n_degree=32 --n_layer=2 --train_time_encoding=learn --edge_embedding=concat --neigh_agg=rnn --path_agg=mean --bias=1e-7 --seed=0 --walk_type=before --direct=add --co_occ=learn_add --gpu=0
```

## Requirements
The experiments ran on NVIDIA RTX A6000 GPUs with 48GB memory and 256GB RAM, using Pytorch 2.0.1 on Ubuntu 22.04 OS. 
The required packages are as follows:
- ```tqdm==4.65.0```
- ```numpy==1.24.3```   
- ```pandas==2.0.3```
- ```numba==0.57.1```
- ```wandb==0.15.10```
- ```python==3.8.18```
- ```pytorch==2.0.1```
- ```pytorch-scatter==2.1.1```


## Cite
  ```
  @inproceedings{kim2024polardsn,
  title={PolarDSN: An Inductive Approach to Learning the Evolution of Network Polarization in Dynamic Signed Networks},
  author={Kim, Min-Jeong and Lee, Yeon-Chang and Kim, Sang-Wook},
  booktitle={Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
  pages={1099--1109},
  year={2024}
}
  ```


