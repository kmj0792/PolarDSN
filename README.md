# PolarDSN-CIKM2024

Bitcoin-alpha
python main.py --data=bitcoinalpha --bs=64 --lr=0.001 --n_degree=64 --n_layer=3 --train_time_encoding=learn --edge_embedding=concat --neigh_agg=rnn --path_agg=mean --bias=1e-6 --seed=0 --walk_type=before --direct=add --co_occ=learn_add --gpu=0

Bitcoin-otc
python main.py --data=bitcoinotc --bs=64 --lr=0.001 --n_degree=128 --n_layer=5 --train_time_encoding=nonlearn --edge_embedding=concat --neigh_agg=rnn --path_agg=mean --bias=1e-6 --seed=0 --walk_type=before --direct=add --co_occ=learn_add --gpu=0

Wiki-RfA
python main.py --data=wiki-RfA --bs=64 --lr=0.001 --n_degree=64 --n_layer=2 --train_time_encoding=nonlearn --edge_embedding=concat --neigh_agg=rnn --path_agg=mean --bias=1e-7 --seed=0 --walk_type=before --direct=add --co_occ=learn_add --gpu=0

Epinions
python main.py --data=epinions --bs=64 --lr=0.001 --n_degree=32 --n_layer=2 --train_time_encoding=learn --edge_embedding=concat --neigh_agg=rnn --path_agg=mean --bias=1e-7 --seed=0 --walk_type=before --direct=add --co_occ=learn_add --gpu=0
