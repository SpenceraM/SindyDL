import os
import torch
from tqdm import tqdm
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import AutoEncoder, compound_loss


def train(train_data, val_data, cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # not used for debugging
    model = AutoEncoder(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    criterion = compound_loss

    val_dict = create_feed_dictionary(val_data, cfg, idxs=None)
    validation_losses = []

    n_batches = cfg['epoch_size']//cfg['batch_size']
    for epoch_idx in tqdm(range(cfg['max_epochs'])):
        for batch_idx in range(n_batches):
            # Zero your gradients for every batch!
            optimizer.zero_grad()

            idxs4batch = np.arange(batch_idx * cfg['batch_size'], (batch_idx + 1) * cfg['batch_size'])
            train_dict = create_feed_dictionary(train_data, cfg, idxs=idxs4batch)
            out = model(train_dict['x:0'].to(device), train_dict['dx:0'].to(device))

            loss, loss_refinement, losses = criterion(out, cfg)
            loss.backward()
            optimizer.step()

        if epoch_idx % cfg['val_frequency'] == 0:
            # run on validation data
            del out, loss, losses, loss_refinement
            model.eval()
            val_out = model(val_dict['x:0'].to(device), val_dict['dx:0'].to(device))
            val_loss, val_loss_refinement, val_losses = criterion(val_out, cfg)
            model.train()
            if cfg['print_progress']:
                print_progress(epoch_idx, val_loss.item(), val_losses, val_out)
            del val_loss, val_losses, val_loss_refinement, val_out

            # Threshold the coefficient mask
        if cfg['sequential_thresholding'] and (epoch_idx % cfg['threshold_frequency'] == 0) and (epoch_idx > 0):
            model.threshold_mask()

    # Refinement stage
    print('REFINEMENT')
    for epoch_idx in tqdm(range(cfg['refinement_epochs'])):
        for batch_idx in range(n_batches):
            # Zero your gradients for every batch!
            optimizer.zero_grad()

            idxs4batch = np.arange(batch_idx * cfg['batch_size'], (batch_idx + 1) * cfg['batch_size'])
            train_dict = create_feed_dictionary(train_data, cfg, idxs=idxs4batch)
            out = model(train_dict['x:0'].to(device), train_dict['dx:0'].to(device))

            loss, loss_refinement, losses = criterion(out, cfg)
            loss_refinement.backward()
            optimizer.step()

        if epoch_idx % cfg['val_frequency'] == 0:
            # run on validation data
            del out, loss, losses, loss_refinement
            model.eval()
            val_out = model(val_dict['x:0'].to(device), val_dict['dx:0'].to(device))
            val_loss, val_loss_refinement, val_losses = criterion(val_out, cfg)
            model.train()
            if cfg['print_progress']:
                print_progress(epoch_idx, val_loss_refinement.item(), val_losses, val_out)
                print(model.coefficient_mask.detach().cpu().numpy())
                print(model.sindy_coefficients.detach().cpu().numpy())
                print()
                print()

def create_feed_dictionary(data, cfg, idxs=None):
    """
    Create the feed dictionary for passing into tensorflow.

    Arguments:
        data - Dictionary object containing the data to be passed in. Must contain input data x,
        along the first (and possibly second) order time derivatives dx (ddx).
        cfg - Dictionary object containing model and training parameters. The relevant
        parameters are model_order (which determines whether the SINDy model predicts first or
        second order time derivatives), sequential_thresholding (which indicates whether or not
        coefficient thresholding is performed), coefficient_mask (optional if sequential
        thresholding is performed; 0/1 mask that selects the relevant coefficients in the SINDy
        model), and learning rate (float that determines the learning rate).
        idxs - Optional array of indices that selects which examples from the dataset are passed
        in to tensorflow. If None, all examples are used.

    Returns:
        feed_dict - Dictionary object containing the relevant data to pass to tensorflow.
    """
    if idxs is None:
        idxs = np.arange(data['x'].shape[0])
    feed_dict = {}
    feed_dict['x:0'] = torch.from_numpy(data['x'][idxs]).float()
    feed_dict['dx:0'] = torch.from_numpy(data['dx'][idxs]).float()
    if cfg['model_order'] == 2:
        feed_dict['ddx:0'] = data['ddx'][idxs]
    if cfg['sequential_thresholding']:
        feed_dict['coefficient_mask:0'] = cfg['coefficient_mask']
    feed_dict['learning_rate:0'] = cfg['learning_rate']
    return feed_dict


def print_progress(epoch_idx, loss, losses, net_output):
    nnz = torch.nonzero(net_output['coefficient_mask'].detach().cpu()).shape[0]
    mask_size = torch.numel(net_output['coefficient_mask'].detach().cpu())
    print('\nEpoch {epoch_idx}: Total Loss {loss:.10f}, NumEl {nnz}/{mask_size}'.format(epoch_idx=epoch_idx, loss=loss,nnz=nnz, mask_size=mask_size))
    for key in losses.keys():
        print('    {key} loss: {loss_val:.10f}'.format(key=key, loss_val=losses[key].item()))
    print()