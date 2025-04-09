python main.py --dataset=Cora --learning_rate=0.01 --gcn_wd=1e-3 --weight_decay=5e-5 --gcn_type=1 --gcn_layers=2  --alpha=0.7 --tau=0.3
python main.py --dataset=CiteSeer --learning_rate=5e-3 --gcn_wd=1e-2 --weight_decay=5e-5 --gcn_type=1 --gcn_layers=2   --alpha=0.8 --tau=0.7
python main.py --dataset=PubMed --learning_rate=5e-3 --gcn_wd=1e-3 --weight_decay=1e-3 --gcn_type=1 --gcn_layers=2   --alpha=0.7 --tau=0.3
python main.py --dataset=film --learning_rate=5e-2 --gcn_wd=1e-4 --weight_decay=1e-3 --gcn_type=1 --gcn_layers=2   --alpha=0.7 --tau=0.9
python main.py --dataset=deezer --learning_rate=0.01 --gcn_wd=1e-3 --weight_decay=5e-4 --gcn_type=1 --gcn_layers=2   --alpha=0.8 --tau=0.9

python main.py --learning_rate=1e-3 --weight_decay=0. --dataset=ogbn-arxiv --gcn_use_bn --gcn_type=2 --gcn_layers=3   --alpha=0.9 --tau=0.9

python main.py --dataset=ogbn-products --learning_rate=5e-4 --weight_decay=0. --gcn_type=2 \
      --gcn_layers=3 --gcn_use_bn 2  --batch_size=150000 \
      --alpha=0.9 --tau=0.7