import os
import argparse

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from models import VAE
from utils.logger import create_logger
from utils.dataloader import create_dataloader
from utils.utils import seed_everything, AverageMeter


def train(args, logger):
    # choose device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # create dataloader
    data_dict = create_dataloader(args, mode='gen')
    train_loader = data_dict['train_loader']
    # test_loader = data_dict['test_loader']

    # create model
    model = VAE(input_dim=52, latent_dim=16, hidden_dim=64)
    model = model.to(device)

    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

    best_loss = 1e6
    loss_meter = AverageMeter('Accuracy')
    # train model
    for epoch in range(args.epochs):
        loss_meter.reset()
        for batch_idx, input_ in enumerate(train_loader):
            input_ = input_.to(device)
            
            recon, mu, log_var = model(input_)
            loss = model.loss_function(recon, input_, mu, log_var)['loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.detach().cpu().numpy())

        if loss_meter.avg < best_loss:
            best_loss = loss_meter.avg
            torch.save(model.state_dict(), 'logs/model.pt')
        logger.info(f'Epoch {epoch}: Loss {loss_meter.avg}')

def parse_args():
    parser = argparse.ArgumentParser(description='Train a VAE')
    # parser.add_argument('--config', type=str, default='configs/vae.yaml', help='Path to the config file.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--data_dir', type=str, default='data/', help='Path to the data directory.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader.')
    return parser.parse_args()

def main():
    args = parse_args()
    logger = create_logger(f"logs/train_vae.log")
    logger.info(args)
    seed_everything(args.seed)
    train(args, logger)

if __name__ == '__main__':
    main()