import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

best_acc = 0
patience = 0

def main(epochs : int, device : torch.device, resnet_version : str = "resnet18"):
    # 1. Konfiguracja urządzenia


    # 2. Przygotowanie danych (STL-10 ma 96x96 px)
    transform_train = transforms.Compose([
        transforms.RandomCrop(96, padding=12),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
    ])

    # Ładowanie zbioru STL-10
    trainset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=96, shuffle=True, num_workers=2)

    testset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=96, shuffle=False, num_workers=2)


    # 3. Definicja i Modyfikacja Modelu
    def get_modified_resnet():
        # Pobieramy standardowy model ResNet (weights=None oznacza trening od zera)
        model = resnet18(weights = None)

        # --- PRZYKŁAD MODYFIKACJI ---

        # mod1: zmiana liczby wyjść w warstwie fc na 10 (dla STL-10)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)

        # mod 2
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # mod 3
        # model.maxpool = nn.Identity()
        model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        return model


    model = get_modified_resnet().to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=0.015, momentum=0.9, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


    # 5. Pętla treningowa (uproszczona)

    def train(epoch):
        e_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        acc = 100. * correct / total
        e_end = time.time()
        print(f'Epoch {epoch} | Loss: {running_loss / len(trainloader):.4f} | Acc: {acc:.2f}% | time: {e_end - e_start:.2f}s')
        #print(f'Epoch {epoch} | Loss: {running_loss / len(trainloader):.4f} | Acc: {acc:.2f}%')

    def test(model, device, test_loader, criterion):
        model.eval()  # Przełącza model w tryb ewaluacji (wyłącza Dropout i BatchNorm)
        test_loss = 0
        correct = 0

        with torch.no_grad():  # Wyłączenie śledzenia gradientów
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                # Sumowanie straty i poprawnych trafień
                test_loss += criterion(outputs, labels).item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()

        test_loss /= len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)

        print(
            f'\n[TEST {resnet_version} | {epoch} / {epochs}] Średnia strata: {test_loss:.4f}, Dokładność: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
        return accuracy, test_loss

    # Uruchomienie na kilka epok
    for epoch in range(epochs):
        train(epoch)
        if epoch % 3 == 0:
            test(model, device, testloader, criterion)

    print("Trening zakończony.")

    # Testowanie modelu
    test(model, device, testloader, criterion)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")
    main(80, device, "resnet18")


# TODO - early stopping
# make this shit more palatable
# unified architecture for the models and running them - classes and polymorphism?