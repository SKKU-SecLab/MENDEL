import os
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import mltk
from typing import Optional
import random
import pickle
import datetime
import sys
import time
import os
from TaPR_pkg import etapr
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # run the code on a specified GPU
from inter_utils import get_data, get_sliding_window_data_flow, get_data_dim
from eval_methods import get_best_f1, get_adjusted_composite_metrics
from usad import *
from utils import *
from sklearn.metrics import *
# TPR, FPR 계산식 TPR = TP/(FP+FN) FPR = FP(FP+TN)

def put_labels(distance, threshold):
    xs = np.zeros_like(distance)
    xs[distance > threshold] = 1
    return xs

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

#from baseline.usad.usad import usad, utils
from . import *
class ModelConfig(mltk.Config):
    x_dim: int = -1
    z_dim: int = 19
    window_length = 15


class TrainConfig(mltk.Config):
    # training params
    batch_size = 200
    max_epoch = 60
    train_start = 0
    max_train_size = None  # `None` means full train set

    early_stopping = True
    valid_portion = 0.3

    save_test_stats = True

    transfer = False
    transfer_type = None
    transfer_size = None
    transfer_path = None

class PredictConfig(mltk.Config):
    load_model_dir: Optional[str]
    test_only = False

    # evaluation params
    test_batch_size = 100
    test_start = 0
    max_test_size = None  # `None` means full test set

    test_hit_rate = False

    save_results = True
    output_dirs = 'analysis_results'
    train_score_filename = 'train_score.pkl'
    test_score_filename = 'test_score.pkl'
    pretrain_test_score_filename = 'pretrain_test_score.pkl'
    preserve_feature_dim = False  # whether to preserve the feature dim in score. If `True`, the score will be a 2-dim ndarray
    anomaly_score_calculate_latency = 1   # How many scores are averaged for the final score at a timestamp. `1` means use last point in each sliding window only.


class ExpConfig(mltk.Config):
    seed = int(time.time())

    dataset ='SWaT'#'omi-1'#

    # model params
    model = ModelConfig()

    # train params
    train = TrainConfig()

    # we do not downsample the data, thus use longer window length according to the params provided in usad paper.
    @mltk.root_checker()
    def _train_post_checker(self, v: 'ExpConfig'):
        if v.model.x_dim == -1:
            v.model.x_dim = get_data_dim(v.dataset, v.train.transfer_size)
        if v.dataset.startswith('machine'):
            v.model.window_length = 25
            v.model.z_dim = 38
            v.train.max_epoch = 250
        if v.dataset == 'SWaT':
            v.model.window_length = 60
            v.model.z_dim = 40
            v.train.max_epoch = 70
        if v.dataset == 'WADI':
            v.model.window_length = 60#5
            v.model.z_dim = 40
            v.train.max_epoch = 70
        if v.dataset == 'HAI':
            v.model.window_length = 10#60
            v.model.z_dim = 20#40
            v.train.max_epoch = 1#70
    test = PredictConfig()        


def main(exp: mltk.Experiment[ExpConfig], config: ExpConfig):
    logging.basicConfig(
        level='INFO',
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    # print the current seed and generate three seeds
    logging.info('Current random seed: %s', config.seed)
    np.random.seed(config.seed)
    random.seed(np.random.randint(0xff))
    np.random.seed(np.random.randint(0xff))

    # print the config
    print(mltk.format_key_values(config, title='Configurations'))
    print('')

    # open the result object and prepare for result directories
    exp.make_dirs('train_summary')
    exp.make_dirs('result_params')
    exp.make_dirs('ckpt_params')
    exp.make_dirs(config.test.output_dirs)
    now = datetime.datetime.now()
    device = get_default_device()
    print(config.train.transfer_type, ' ', config.train.transfer_size)
    # simple data
    (x_train, _), (x_test, y_test) = \
        get_data(config.dataset, config.train.max_train_size, config.test.max_test_size,
                 train_start=config.train.train_start, test_start=config.test.test_start,
                 valid_portion=config.train.valid_portion, prefix=config.train.transfer_path, size=config.train.transfer_size, tr_type=config.train.transfer_type)#prefix='./input/SWaT/')##prefix="./input/WADI/SWaT-WADI/")#"./input/SWaT/SWaT-WADI/")#"./input/SWaT/WADI-SWaT/")#"./input/WADI/WADI-SWaT/")#
    split_idx = int(len(x_train) * config.train.valid_portion)
    x_train, x_valid = x_train[:-split_idx], x_train[-split_idx:]

    train_data = get_sliding_window_data_flow(window_size=config.model.window_length,
                                              batch_size=config.train.batch_size,
                                              x=x_train, shuffle=False, skip_incomplete=False).get_arrays()[0]

    val_data = get_sliding_window_data_flow(window_size=config.model.window_length,
                                              batch_size=config.train.batch_size,
                                              x=x_valid, shuffle=False, skip_incomplete=False).get_arrays()[0]

    test_data = get_sliding_window_data_flow(window_size=config.model.window_length,
                                             batch_size=config.test.test_batch_size,
                                             x=x_test, shuffle=False, skip_incomplete=False).get_arrays()[0]

    w_size = config.model.window_length * config.model.x_dim
    z_size = config.model.window_length * config.model.z_dim

    train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(train_data).float().view([train_data.shape[0], w_size])
    ), batch_size=config.train.batch_size, shuffle=False, num_workers=0)

    val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(val_data).float().view([val_data.shape[0], w_size])
    ), batch_size=config.train.batch_size, shuffle=False, num_workers=0)

    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(test_data).float().view([test_data.shape[0], w_size])
    ), batch_size=config.test.test_batch_size, shuffle=False, num_workers=0)

    model = UsadModel(w_size, z_size)
    model = to_device(model, device)

    if config.test.test_only:
        checkpoint = torch.load("model_SWaT.pth")
    
    elif config.train.transfer:
        checkpoint = torch.load(f"model_" + "WADI" + "_" + str(config.train.transfer_size) + "_" + config.train.transfer_type + ".pth")
        model.encoder.load_state_dict(checkpoint['encoder'])
        model.decoder1.load_state_dict(checkpoint['decoder1'])
        model.decoder2.load_state_dict(checkpoint['decoder2'])

        start_time = time.time()
        history = training(config.train.max_epoch, model, train_loader, val_loader)
        train_time = time.time() - start_time
        torch.save({
            'encoder': model.encoder.state_dict(),
            'decoder1': model.decoder1.state_dict(),
            'decoder2': model.decoder2.state_dict()
        }, f"model_transfer_WS_" + str(config.train.transfer_size) + "_" + config.train.transfer_type + ".pth")
        
        checkpoint = torch.load(f"model_transfer_WS_" + str(config.train.transfer_size) + "_" + config.train.transfer_type + ".pth")  
    else:
        start_time = time.time()
        history = training(config.train.max_epoch, model, train_loader, val_loader)
        train_time = time.time() - start_time
        torch.save({
            'encoder': model.encoder.state_dict(),
            'decoder1': model.decoder1.state_dict(),
            'decoder2': model.decoder2.state_dict()
        }, f"model_" + "WADI" + "_" + str(config.train.transfer_size) + "_" + config.train.transfer_type + ".pth")#}, #"model_WADI.pth")#
        checkpoint = torch.load(f"model_" + "WADI" + "_" + str(config.train.transfer_size) + "_" + config.train.transfer_type + ".pth")#("model_WADI.pth")#
    
        #checkpoint = torch.load("model_SWaT.pth")#("model_SWaT.pth")
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder1.load_state_dict(checkpoint['decoder1'])
    model.decoder2.load_state_dict(checkpoint['decoder2'])

    start_time = time.time()
    results = testing(model, test_loader)

    print("===========================anomaly score extracted===========================")
    test_time =  time.time() - start_time 
    test_score = -np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                             results[-1].flatten().detach().cpu().numpy()])  # negative score, lower means more anomalous
    print("test score shape", test_score.shape)
    assert len(test_score) <= len(y_test)

    with open(os.path.join(exp.abspath(config.test.output_dirs), config.test.test_score_filename), 'wb') as file:
        pickle.dump(test_score, file)

    len_y_test = len(y_test)    # original label length
    y_test = y_test[-len(test_score):]
 
    t, th = get_best_f1(test_score, y_test)
    # output the results
    exp.update_results({
        'best-f1': t[0],
        'precision': t[1],
        'recall': t[2],
        'TP': t[3],
        'TN': t[4],
        'FP': t[5],
        'FN': t[6],
        'latency': 'TODO',
        'threshold': th
    })

 
    print('TP num: ', t[3])
    tp_idx = np.logical_and(y_test > 0.5, test_score <= th)
    tp_idx = np.where(tp_idx)[0]
    print('tp_idx len:', len(tp_idx))
    tp_idx += (len_y_test - len(test_score))


    auroc, ap, _, _, _, _, _ = get_adjusted_composite_metrics(test_score, y_test)
    exp.update_results({
        'auroc': auroc,
        'ap': ap
    })

    # usad does not have test score for each dim, cannot calculate IPS score.

    print('')
    print(mltk.format_key_values(exp.results), 'Results')
    print(now)
    print("training time ", train_time)
    print("testing time ", test_time)
    
if __name__ == '__main__':
    with mltk.Experiment(ExpConfig()) as exp:
        exp.save_config()
        main(exp, exp.config)
