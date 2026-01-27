import csv
import datetime
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
from torchvision.models import resnet18

from scripts.img_proto import get_stl10_class_weights_learn_difficult, get_stl10_class_weights_learn_easy_linear, get_stl10_class_weights_learn_easy_inversion


def save_results_to_csv(results, csv_file_path):
    """
    Save the output of main() function to a CSV file.

    Args:
        results: List of results for each fold.
                Each element is a tuple of (baseline_loss, baseline_acc, weighted_loss, weighted_acc)
                where each is a list of 10 values (one per class).
        csv_file_path: Path to the output CSV file.
    """
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['fold', 'class_idx', 'baseline_loss', 'baseline_acc', 'weighted_loss', 'weighted_acc']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for fold_idx, (baseline_loss_by_class, baseline_acc_by_class, weighted_loss_by_class, weighted_acc_by_class) in enumerate(results):
            for class_idx in range(10):
                writer.writerow({
                    'fold': fold_idx,
                    'class_idx': class_idx,
                    'baseline_loss': baseline_loss_by_class[class_idx],
                    'baseline_acc': baseline_acc_by_class[class_idx],
                    'weighted_loss': weighted_loss_by_class[class_idx],
                    'weighted_acc': weighted_acc_by_class[class_idx]
                })


def get_modified_resnet():
    model = resnet18(weights=None)

    # make it compatible with STL-10 (10 classes)
    num_input_features = model.fc.in_features
    model.fc = nn.Linear(num_input_features, 10)

    # improve performance on small images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    return model


def main(max_epochs: int, device: torch.device, weighting_scheme: str = "learn_difficult", debug : bool = False):
    print(f"\n--- Running {weighting_scheme} ---\n")
    # full dataset
    full_dataset = torchvision.datasets.STL10(root='./data', split='train', download=True)

    # transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(96, padding=12, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
    ])

    # 10-fold cross-validation
    results = []
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for fold, (train_indices, val_indices) in enumerate(kf.split(full_dataset)):

        # create data loaders for this fold
        training_subset = torch.utils.data.Subset(full_dataset, train_indices)
        validation_subset = torch.utils.data.Subset(full_dataset, val_indices)

        training_subset.dataset.transform = transform_train
        validation_subset.dataset.transform = transform_test

        train_loader = torch.utils.data.DataLoader(training_subset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(validation_subset, batch_size=64, shuffle=False)

        # baseline init
        model_baseline = get_modified_resnet().to(device)
        criterion_baseline = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer_baseline = optim.SGD(model_baseline.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)
        scheduler_baseline = optim.lr_scheduler.CosineAnnealingLR(optimizer_baseline, T_max=max_epochs)

        # weighted init
        weights = None
        if weighting_scheme == "learn_easy_inversion":
            weights = get_stl10_class_weights_learn_easy_inversion().to(device)
        elif weighting_scheme == "learn_easy_linear":
            weights = get_stl10_class_weights_learn_easy_linear().to(device)
        else:
            weights = get_stl10_class_weights_learn_difficult().to(device)

        model_weighted = get_modified_resnet().to(device)
        criterion_weighted = nn.CrossEntropyLoss(label_smoothing=0.1, weight=weights)
        optimizer_weighted = optim.SGD(model_weighted.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)
        scheduler_weighted = optim.lr_scheduler.CosineAnnealingLR(optimizer_weighted, T_max=max_epochs)

        # training loop
        for epoch in range(max_epochs):
            # set models to training mode
            model_baseline.train()
            model_weighted.train()

            # initialize metrics
            running_loss_baseline = 0.0
            total_correct_baseline = 0
            total_baseline = 0

            running_loss_weighted = 0.0
            total_correct_weighted = 0
            total_weighted = 0

            for i, (inputs, labels) in enumerate(train_loader):
                # transfer data to device
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer_baseline.zero_grad()
                optimizer_weighted.zero_grad()

                # compute outputs
                outputs_baseline = model_baseline(inputs)
                outputs_weighted = model_weighted(inputs)

                # compute and propagate loss
                total_loss_baseline = criterion_baseline(outputs_baseline, labels)
                total_loss_baseline.backward()

                total_loss_weighted = criterion_weighted(outputs_weighted, labels)
                total_loss_weighted.backward()

                # update weights
                optimizer_baseline.step()
                optimizer_weighted.step()

                # update metrics
                running_loss_baseline += total_loss_baseline.item()
                _, predicted = outputs_baseline.max(1)
                total_baseline += labels.size(0)
                total_correct_baseline += predicted.eq(labels).sum().item()

                running_loss_weighted += total_loss_weighted.item()
                _, predicted = outputs_weighted.max(1)
                total_weighted += labels.size(0)
                total_correct_weighted += predicted.eq(labels).sum().item()

            # update learning rate
            scheduler_baseline.step()
            training_acc_baseline = total_correct_baseline / total_baseline
            training_avg_loss_baseline = running_loss_baseline / len(train_loader)

            scheduler_weighted.step()
            training_acc_weighted = total_correct_weighted / total_weighted
            training_avg_loss_weighted = running_loss_weighted / len(train_loader)

            print(f"Fold {fold} | Epoch {epoch}/{max_epochs}:")
            if debug:
                print(f"\tBaseline: Loss: {training_avg_loss_baseline:.4f} | Acc: {training_acc_baseline * 100:.2f}%")
                print(f"\tWeighted: Loss: {training_avg_loss_weighted:.4f} | Acc: {training_acc_weighted * 100:.2f}%")

        # fold evaluation
        model_baseline.eval()
        model_weighted.eval()

        # element-wise evaluation criterion
        test_criterion = nn.CrossEntropyLoss(reduction="none")

        avg_acc_by_class_baseline = [0] * 10
        avg_loss_by_class_baseline = [0.0] * 10

        avg_acc_by_class_weighted = [0] * 10
        avg_loss_by_class_weighted = [0] * 10

        total_by_class = [0] * 10

        with (torch.no_grad()):
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # baseline evaluation
                outputs_baseline = model_baseline(inputs)
                _, predicted_baseline = outputs_baseline.max(1)

                # weighted evaluation
                outputs_weighted = model_weighted(inputs)
                _, predicted_weighted = outputs_weighted.max(1)

                losses_baseline = test_criterion(outputs_baseline, labels)
                losses_weighted = test_criterion(outputs_weighted, labels)

                for i in range(len(labels)):
                    label = labels[i].item()
                    total_by_class[label] += 1

                    # baseline metrics
                    avg_loss_by_class_baseline[label] += losses_baseline[i].item()
                    if predicted_baseline[i].item() == labels[i]:
                        avg_acc_by_class_baseline[label] += 1

                    # weighted metrics
                    avg_loss_by_class_weighted[label] += losses_weighted[i].item()
                    if predicted_weighted[i].item() == labels[i]:
                        avg_acc_by_class_weighted[label] += 1

        avg_acc_by_class_baseline = [correct / total if total > 0 else 0 for correct, total in
                                     zip(avg_acc_by_class_baseline, total_by_class)]
        avg_loss_by_class_baseline = [loss / total if total > 0 else 0 for loss, total in
                                      zip(avg_loss_by_class_baseline, total_by_class)]

        avg_acc_by_class_weighted = [correct / total if total > 0 else 0 for correct, total in
                                     zip(avg_acc_by_class_weighted, total_by_class)]
        avg_loss_by_class_weighted = [loss / total if total > 0 else 0 for loss, total in
                                      zip(avg_loss_by_class_weighted, total_by_class)]

        if debug:
            print(f"Fold {fold} Evaluation Results:")
            for class_idx in range(10):
                print(f"\tClass {class_idx}:")
                print(f"\t\tBaseline - Loss: {avg_loss_by_class_baseline[class_idx]:.4f} | Acc: {avg_acc_by_class_baseline[class_idx]*100:.2f}%")
                print(f"\t\tWeighted - Loss: {avg_loss_by_class_weighted[class_idx]:.4f} | Acc: {avg_acc_by_class_weighted[class_idx]*100:.2f}%")

        results.append((avg_loss_by_class_baseline, avg_acc_by_class_baseline, avg_loss_by_class_weighted, avg_acc_by_class_weighted))
    return results

if __name__ == "__main__":
    start = datetime.datetime.now()
    print(f"Start: {start}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")
    epochs = 40

    results_learn_difficult = main(epochs, device, weighting_scheme="learn_difficult")
    save_results_to_csv(results_learn_difficult, 'results_learn_difficult.csv')

    results_learn_easy_inversion = main(epochs, device, weighting_scheme="learn_easy_inversion")
    save_results_to_csv(results_learn_easy_inversion, 'results_learn_easy_inversion.csv')

    results_learn_easy_linear = main(epochs, device, weighting_scheme="learn_easy_linear")
    save_results_to_csv(results_learn_easy_linear, 'results_learn_easy_linear.csv')

    end = datetime.datetime.now()
    print(f"End: {end}")
    print(f"Elapsed: {(end - start).total_seconds() // 60} minutes")


