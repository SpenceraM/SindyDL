import sys
import os
import argparse
import yaml
import datetime
import pandas as pd
import numpy as np
sys.path.append("../../src")
from example_lorenz import get_lorenz_data
from sindy_utils import library_size
from trainer import train
# from training import train_network
# import tensorflow as tf


if __name__ == '__main__':
    np.random.seed(0)
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

    # generate training, validation, testing data
    print('Generating training data')
    training_data = get_lorenz_data(cfg.get('n_train_ics',1024), noise_strength=cfg.get('noise_strength', 1e-6))
    print('Generating validation data')
    validation_data = get_lorenz_data(cfg.get('n_val_ics',20), noise_strength=cfg.get('noise_strength', 1e-6))
    print('Finished generating data\n')
    # Finish setting up model and training parameters
    cfg['library_dim'] = library_size(cfg['latent_dim'], cfg['poly_order'], cfg['include_sine'], True)
    cfg['coefficient_mask'] = np.ones((cfg['library_dim'], cfg['latent_dim']))
    cfg['epoch_size'] = training_data['x'].shape[0]
    cfg['data_path'] = os.getcwd() + '/'

    # Run training experiment
    df = pd.DataFrame()
    for i in range(cfg['num_experiments']):
        print('EXPERIMENT %d' % i)
        cfg['coefficient_mask'] = np.ones((cfg['library_dim'], cfg['latent_dim']))
        cfg['save_name'] = 'lorenz_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

        results_dict = train(training_data, validation_data, cfg)
        print()
        # df = df.append({**results_dict, **params}, ignore_index=True)


    df.to_pickle('experiment_results_' + datetime.datetime.now().strftime("%Y%m%d%H%M") + '.pkl')