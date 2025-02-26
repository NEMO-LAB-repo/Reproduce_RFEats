import os
import shutil
import argparse

import torch
import numpy as np
from tqdm import tqdm

from models import VAE
from utils.utils import seed_everything


def generate(args):
    # choose device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # create model
    model = VAE(input_dim=52, latent_dim=16, hidden_dim=64)
    model.load_state_dict(torch.load('logs/model.pt'))
    model = model.to(device)

    # remove old data
    generated_dir = 'data/generate'
    if os.path.exists(generated_dir):
        shutil.rmtree(generated_dir)
    os.mkdir(generated_dir)

    # generate data
    for i in tqdm(range(args.num_generated)):
        with torch.no_grad():
            sample = model.sample(1, device).squeeze().cpu().numpy()
        data = np.split(sample, 2, axis=0)
        # data = torch.split(sample, 26, dim=1)
        with open(os.path.join(generated_dir, f'{i}.txt'), 'w') as f:
            for freq, amplitude, phase in zip(range(500, 1020, 20), data[0], data[1]):
                f.write(f'{freq},{amplitude},{phase}\n')
                # print(f'write: {freq},{amplitude*float(los_amplitude)},{phase*float(los_phase)}')

def parse_args():
    parser = argparse.ArgumentParser(description='Train a VAE')
    # parser.add_argument('--config', type=str, default='configs/vae.yaml', help='Path to the config file.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--data_dir', type=str, default='data/', help='Path to the data directory.')
    parser.add_argument('--num_generated', type=int, default=64, help='Number of samples to generate.')
    return parser.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)
    generate(args)

if __name__ == '__main__':
    main()