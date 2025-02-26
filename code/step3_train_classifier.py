import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix

from models import Classifier
from utils.logger import create_logger
from utils.dataloader import create_dataloader
from utils.utils import seed_everything, AverageMeter


def train(args, logger):
    # choose device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # create dataloader
    data_dict = create_dataloader(args, mode='cls')
    train_loader = data_dict['train_loader']
    test_loader = data_dict['test_loader']
    logger.info(len(data_dict['train_dataset']))

    # create model
    model = Classifier(input_dim=52, hidden_dim=64, num_classes=args.num_classes)
    model = model.to(device)

    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0
    # train model
    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (input_, label) in enumerate(train_loader):
            input_ = input_.to(device)
            label = label.to(device)
            
            outputs = model(input_)
            loss = criterion(outputs, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}: Loss {loss.item()}')

        if (epoch + 1) % 5 == 0:
            results = test(model, test_loader, device)
            logger.info(f"Accuracy: {results['accuracy']}")
            logger.info(f"Confusion Matrix: \n{results['confusion_matrix']}")
            
            if results['accuracy'] > best_acc:
                best_acc = results['accuracy']
                torch.save(model.state_dict(), 'logs/classifier.pt')

    # test using best model
    model.load_state_dict(torch.load('logs/classifier.pt'))
    results = test(model, test_loader, device)
    logger.info(f"Final Accuracy: {results['accuracy']}")
    logger.info(f"Confusion Matrix: \n{results['confusion_matrix']}")

def test(model, test_loader, device):
    model.eval()
    acc = AverageMeter('Accuracy')
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, (input_, label) in enumerate(test_loader):
            input_ = input_.to(device)
            label = label.to(device)

            outputs = model(input_)
            _, pred = torch.max(outputs, 1)
            
            correct = (pred == label).sum().item()
            total = label.size(0)
            acc.update(correct / total, total)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    return {
        'accuracy': acc.avg,
        'confusion_matrix': confusion_matrix(np.array(all_labels), np.array(all_preds)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Train a VAE')
    # parser.add_argument('--config', type=str, default='configs/vae.yaml', help='Path to the config file.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--gpus', type=str, default='0', help='GPU to use.')
    parser.add_argument('--data_dir', type=str, default='data/', help='Path to the data directory.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader.')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of data classes.')
    parser.add_argument('--use-generator', action='store_true', help='Use generated data.')
    return parser.parse_args()

def main():
    args = parse_args()
    logger = create_logger(f"logs/train_classifier{'_generate' if args.use_generator else ''}.log")
    logger.info(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    seed_everything(args.seed)
    train(args, logger)

if __name__ == '__main__':
    main()