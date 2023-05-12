import sys
sys.path.append("../../src")
import os
import argparse
import yaml

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import datetime
import pandas as pd
import numpy as np
from example_lorenz import get_lorenz_data
# from sindy_utils import library_size
# from training import train_network
# import tensorflow as tf

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

    # generate training, validation, testing data
    print('Generating training data')
    training_data = get_lorenz_data(cfg.get('n_train_ics',1000), noise_strength=cfg.get('noise_strength', 1e-6))
    print('Generating validation data')
    validation_data = get_lorenz_data(cfg.get('n_val_ics',100), noise_strength=cfg.get('noise_strength', 1e-6))

    print()