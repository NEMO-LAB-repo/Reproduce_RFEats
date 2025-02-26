import os
import argparse
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from models import Classifier
from utils.logger import create_logger
from utils.dataloader import create_dataloader
from utils.utils import seed_everything, AverageMeter


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
        'confusion_matrix': confusion_matrix(np.array(all_labels), np.array(all_preds))
    }


def main():
    parser = argparse.ArgumentParser(description='Classify with a pre-trained model')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--gpus', type=str, default='0', help='GPU to use.')
    parser.add_argument('--data_dir', type=str, default='data/', help='Path to the data directory.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader.')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of data classes.')
    parser.add_argument('--use-generator', action='store_true', help='Use generated data if available.')
    args = parser.parse_args()

    logger = create_logger("logs/classify.log")
    logger.info(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    seed_everything(args.seed)

    # Load the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Classifier(input_dim=52, hidden_dim=64, num_classes=args.num_classes)
    model_state = torch.load('logs/classifier.pt')
    model.load_state_dict(model_state)
    model = model.to(device)

    # Create dataloader
    data_dict = create_dataloader(args, mode='cls')
    test_loader = data_dict['test_loader']

    # Perform testing
    results = test(model, test_loader, device)
    logger.info(f"Final Accuracy: {results['accuracy']}")
    logger.info(f"Confusion Matrix: \n{results['confusion_matrix']}")


if __name__ == '__main__':
    main()
