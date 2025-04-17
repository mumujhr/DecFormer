import argparse
import torch
import torch.nn.functional as F
import numpy as np
import yaml
import os
import os.path as osp
from yaml import SafeLoader
from Data.get_data import get_data
from run import run
from run_batch import run_batch
from Data.data_utils import load_fixed_splits
import random
import warnings
from Layer.SPDEC import calculate_calibration_mask

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--gcn_wd', type=float, default=5e-4)
    parser.add_argument('--gpu_id', type=int, default=6)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--gcn_use_bn', action='store_true', help='gcn use batch norm')
    parser.add_argument('--show_details', type=bool, default=True)
    parser.add_argument('--gcn_type', type=int, default=1)
    parser.add_argument('--gcn_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=100000)
    parser.add_argument('--rand_split', action='store_true', help='random split dataset')
    parser.add_argument('--rand_split_class', action='store_true', help='random split dataset by class')
    parser.add_argument('--protocol', type=str, default='semi')
    parser.add_argument('--label_num_per_class', type=int, default=20)
    parser.add_argument('--train_prop', type=float, default=.6)
    parser.add_argument('--valid_prop', type=float, default=.2)
    parser.add_argument('--alpha', type=float, default=.8)
    parser.add_argument('--tau', type=float, default=.3)
    parser.add_argument('--lambad_val',type=float,default=0.3)

    args = parser.parse_args()

    assert args.gpu_id in range(0, 8)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    # device = torch.device('cpu')

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]
    fix_seed(config['seed'])

    # path = osp.join(osp.expanduser('~'), 'datasets/')
    # path = '/Users/jihaoran/PycharmProjects/CoBFormer/Data'
    path = '/Users/jihaoran/PycharmProjects/Graduation_Design/datasets'

    results = dict()
    alpha = args.alpha
    tau = args.tau

    # postfix = f'{n_patch}'
    postfix = "test"
    runs = 5

    data = get_data(path, args.dataset)

    # print(data.graph.keys())
    # dict_keys(['edge_index', 'node_feat', 'edge_feat', 'num_nodes'])

    edge_index = data.graph['edge_index']
    num_nodes = data.graph['num_nodes']
    # lambda_val = 0.3

    # data.graph['calibration_mask'] = None

    # load calibration mask
    calibration_mask_path = osp.join(path, args.dataset, 'calibration_mask_'+ args.dataset + '.pt')
    if osp.exists(calibration_mask_path):
        calibration_mask = torch.load(calibration_mask_path)
    else:
        calibration_mask = calculate_calibration_mask(edge_index, num_nodes, args.lambad_val)
        # save calibration mask
        os.makedirs(osp.dirname(calibration_mask_path), exist_ok=True)
        torch.save(calibration_mask, calibration_mask_path)


    # calculate calibration mask
    # step 1: calculate shortest path distance matrix
    # step 2: calculate calibration mask: exp(-distance/tau)

    # calibration_mask = calculate_calibration_mask(edge_index, num_nodes, lambda_val)
    data.graph['calibration_mask'] = None
    data.graph['calibration_mask'] = calibration_mask
    data.graph['calibration_mask'] = data.graph['calibration_mask'].to(device)
    # print(" data.graph['calibration_mask']——shape:", data.graph['calibration_mask'].shape)

    print('calibration_mask Done!!!')
    
    # get the splits for all runs
    if args.rand_split:
        split_idx_lst = [data.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                         for _ in range(runs)]
    elif args.rand_split_class:
        split_idx_lst = [data.get_idx_split(split_type='class', label_num_per_class=args.label_num_per_class)
                         for _ in range(runs)]
    else:
        split_idx_lst = load_fixed_splits(path, data, name=args.dataset, protocol=args.protocol)

    batch_size = args.batch_size

    results = [[], []]
    for r in range(runs):
        if args.dataset in ['Cora', 'CiteSeer', 'PubMed', 'ogbn-arxiv', 'ogbn-products'] and args.protocol == 'semi':
            split_idx = split_idx_lst[0]
        else:
            split_idx = split_idx_lst[r]

        res_gnn, res_trans = run(args, config, device, data, split_idx, alpha, tau, postfix)
        results[0].append(res_gnn)
        results[1].append(res_trans)

    print(f"==== Final GNN====")
    result = torch.tensor(results[0]) * 100.
    print(result)
    print(f"max: {torch.max(result, dim=0)[0]}")
    print(f"min: {torch.min(result, dim=0)[0]}")
    print(f"mean: {torch.mean(result, dim=0)}")
    print(f"std: {torch.std(result, dim=0)}")

    print(f'GNN Micro: {torch.mean(result, dim=0)[1]:.2f} ± {torch.std(result, dim=0)[1]:.2f}')
    print(f'GNN Macro: {torch.mean(result, dim=0)[3]:.2f} ± {torch.std(result, dim=0)[3]:.2f}')

    print(f"==== Final Trans====")
    result = torch.tensor(results[1]) * 100.
    print(result)
    print(f"max: {torch.max(result, dim=0)[0]}")
    print(f"min: {torch.min(result, dim=0)[0]}")
    print(f"mean: {torch.mean(result, dim=0)}")
    print(f"std: {torch.std(result, dim=0)}")

    print(f'Trans Micro: {torch.mean(result, dim=0)[1]:.2f} ± {torch.std(result, dim=0)[1]:.2f}')
    print(f'Trans Macro: {torch.mean(result, dim=0)[3]:.2f} ± {torch.std(result, dim=0)[3]:.2f}')
