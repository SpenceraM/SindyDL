import sys
import os
import argparse
import yaml
import datetime
import pandas as pd
import numpy as np
import torch

sys.path.append("../../src")
from example_pendulum import get_pendulum_data
from sindy_utils import library_size
from trainer import train



if __name__ == '__main__':


    # argparse to load yaml for config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='path to config file')
    args = parser.parse_args()
    cfg_path = args.config

    # load config
    with open(cfg_path, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Finish setting up model and training parameters
    cfg['library_dim'] = library_size(2 * cfg['latent_dim'],  cfg['poly_order'], cfg['include_sine'], True)
    cfg['coefficient_mask'] = np.ones((cfg['library_dim'], cfg['latent_dim']))
    cfg['data_path'] = os.getcwd() + '/'

    # Run training experiment
    df = pd.DataFrame()
    for i in range(cfg['num_experiments']):
        print('EXPERIMENT %d' % i)
        np.random.seed(i)
        torch.manual_seed(i)
        # generate training, validation, testing data
        print('Generating training data')
        training_data = get_pendulum_data(cfg.get('n_train_ics', 100))
        cfg['epoch_size'] = training_data['x'].shape[0]
        print('Generating validation data')
        validation_data = get_pendulum_data(cfg.get('n_val_ics', 10))
        print('Finished generating data\n')

        cfg['coefficient_mask'] = np.ones((cfg['library_dim'], cfg['latent_dim']))
        cfg['save_name'] = 'pendulum_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        cfg['input_dim'] = training_data['x'].shape[-1]

        results_dict = train(training_data, validation_data, cfg)
        results_dict['seed'] = i
        df = df.append({**results_dict, **cfg}, ignore_index=True)
        df.to_pickle('experiment_results_' + datetime.datetime.now().strftime("%Y%m%d%H%M") + '.pkl')