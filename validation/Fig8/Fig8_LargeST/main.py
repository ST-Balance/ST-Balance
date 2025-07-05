import os
import numpy as np

import sys
import networkx as nx
import yaml
from karateclub import GraRep, HOPE
from node2vec import Node2Vec

# from umap import UMAP

sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch

torch.set_num_threads(3)

from src.models.STBalance.stbanlance import STBalance
from src.base.engine import BaseEngine
from src.utils.args import get_public_config
from src.utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info
from src.utils.graph_algo import normalize_adj_mx
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()

    parser.add_argument('--his_len', type=int, default=96 * 14)
    parser.add_argument('--if_enhance', type=int, default=1)
    parser.add_argument('--enhance_dim', type=int, default=12)
    parser.add_argument('--if_en', type=int, default=1)
    parser.add_argument('--if_de', type=int, default=1)
    parser.add_argument('--fusion_num_step', type=int, default=2)
    parser.add_argument('--fusion_num_layer', type=int, default=3)
    parser.add_argument('--fusion_dim', type=int, default=64)
    parser.add_argument('--fusion_out_dim', type=int, default=16)
    parser.add_argument('--fusion_dropout', type=float, default=0.2)
    parser.add_argument('--if_forward', type=int, default=1)
    parser.add_argument('--if_backward', type=int, default=1)
    parser.add_argument('--node_dim', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=1)
    parser.add_argument('--if_T_i_D', type=int, default=1)
    parser.add_argument('--if_D_i_W', type=int, default=1)
    parser.add_argument('--temp_dim_tid', type=int, default=32)
    parser.add_argument('--temp_dim_diw', type=int, default=32)
    parser.add_argument('--if_feedback', type=int, default=1)
    parser.add_argument('--time_of_day_size', type=int, default=96)
    parser.add_argument('--day_of_week_size', type=int, default=7)

    parser.add_argument('--adj_type', type=str, default='doubletransition')
    parser.add_argument('--lrate', type=float, default=2e-3)
    parser.add_argument('--wdecay', type=float, default=1e-4)
    parser.add_argument('--clip_grad_value', type=float, default=5)
    args = parser.parse_args()
    with open(f"./config/ST-Balance.yaml", "r") as f:
        cfg = yaml.safe_load(f)[args.dataset]
    vars(args).update(cfg)
    log_dir = './experiments/{}/{}/'.format(args.model_name, args.dataset)
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)

    return args, log_dir, logger


def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)

    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    logger.info('Adj path: ' + adj_path)

    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = normalize_adj_mx(adj_mx, args.adj_type)
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    dataloader, scaler = load_dataset(data_path, args, logger)

    model = STBalance(node_num=node_num,
                      input_dim=args.input_dim,
                      output_dim=args.output_dim,
                      model_args=vars(args),
                      supports=supports)

    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    steps = [1, 20, 40, 60, 80]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=0.5)

    engine = BaseEngine(device=device,
                        model=model,
                        dataloader=dataloader,
                        scaler=scaler,
                        sampler=None,
                        loss_fn=loss_fn,
                        lrate=args.lrate,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        clip_grad_value=args.clip_grad_value,
                        max_epochs=args.max_epochs,
                        patience=args.patience,
                        log_dir=log_dir,
                        logger=logger,
                        seed=args.seed,
                        )

    if args.mode == 'train':
        engine.train()
    else:
        engine.evaluate(args.mode)


if __name__ == "__main__":
    main()
