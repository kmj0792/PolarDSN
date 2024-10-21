# PolarDSN: An Inductive Approach to Learning the Evolution of Network Polarization in Dynamic Signed Networks

![git_overview](https://github.com/user-attachments/assets/5962613d-ddca-4f50-92ed-cbb8b69e2d89)

This repository provides a reference implementation of PolarDSN as described in the following paper "[PolarDSN: An Inductive Approach to Learning the Evolution of Network Polarization in Dynamic Signed Networks](https://doi.org/10.1145/3627673.3679654)", published at CIKM 2024 (full paper). (33rd ACM International Conference on Information and Knowledge Management (ACM CIKM 2024))

## Authors
- Min-Jeong Kim (kmj0792@hanyang.ac.kr)
- Yeon-Chang Lee (yeonchang@unist.ac.kr)
- Sang-Wook Kim (wook@hanyang.ac.kr)

## Inputs
The input dataset should be saved in ```./DynamicData/weight/``` folder. 
<img src = "https://github.com/user-attachments/assets/88e73895-37e3-4bd2-85e7-e95fe095f955" width="50" height="20">

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

+ Epinions:
```
python main.py --data=epinions --bs=64 --lr=0.001 --n_degree=32 --n_layer=2 --train_time_encoding=learn --edge_embedding=concat --neigh_agg=rnn --path_agg=mean --bias=1e-7 --seed=0 --walk_type=before --direct=add --co_occ=learn_add --gpu=0
```

## Requirements
The code has been tested running under Python 3.7.4. The required packages are as follows:
- ```dgl==0.4.1```
- ```tqdm==4.64.0```
- ```numpy==1.16.4```
- ```pandas==0.25.0```
- ```tqdm==4.64.0```
- ```scipy==1.3.0```
- ```scikit-learn==0.21.2```  
- ```torch-geometric==2.2.0```
- ```torch-scatter==2.1.0+pt112cu116```
- ```torch-sparse==0.6.16+pt112cu116```
- ```conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge```

## Cite
  ```
  
  ```


