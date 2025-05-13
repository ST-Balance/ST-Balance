import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Corrformer for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--model_id', type=str, default='0', help='model id')
parser.add_argument('--model', type=str, default='STBalance',
                    help='model name, options: [STBalance]')

# data loader
parser.add_argument('--data', type=str, default='Global_Wind', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/global_wind/', help='root path of the data file')
parser.add_argument('--pos_filename', type=str, default='./dataset/global_wind/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--test_features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=int, default=0, help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=48, help='input sequence length')
parser.add_argument('--label_len', type=int, default=24, help='start token length')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')

# optimization
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='Exp', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type2', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', type=bool, default=False, help='use multi gpu')
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

# ------
parser.add_argument('--num_nodes', default=3850)
parser.add_argument('--his_len', type=int, default=168)
parser.add_argument('--if_enhance', default=True)
parser.add_argument('--enhance_dim', type=int, default=48)
parser.add_argument('--if_en', default=True)
parser.add_argument('--if_de', default=True)
parser.add_argument('--fusion_num_step', default=2)
parser.add_argument('--fusion_num_layer', default=3)
parser.add_argument('--fusion_dim', default=64)
parser.add_argument('--fusion_out_dim', type=int, default=32)
parser.add_argument('--fusion_dropout', default=0.2)
parser.add_argument('--if_forward', default=True)
parser.add_argument('--if_backward', default=True)
parser.add_argument('--node_dim', default=64)
parser.add_argument('--if_feedback', default=True)
parser.add_argument('--nhead', default=1)
parser.add_argument('--if_time_in_day', default=True)
parser.add_argument('--if_day_in_week', default=True)
parser.add_argument('--if_day_in_month', default=True)
parser.add_argument('--if_day_in_year', default=True)
parser.add_argument('--temp_dim', default=16)
parser.add_argument('--time_of_day_size', default=24)
parser.add_argument('--day_of_week_size', default=7)
parser.add_argument('--day_of_month_size', default=31)
parser.add_argument('--day_of_year_size', default=365)

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
