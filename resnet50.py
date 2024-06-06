import sys
import os
import math
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    for (inputs, labels) in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100
    return accuracy


if __name__ == "__main__":
    # name_mapping = {
    #     "Bulbasaur": "妙蛙种子",
    #     "Charmander": "小火龙",
    #     "Mewtwo": "超梦",
    #     "Pikachu": "皮卡丘",
    #     "Squirtle": "杰尼龟",
    #     "Psyduck": "可达鸭",
    #     "Gengar": "耿鬼",
    #     "Ninetales": "九尾",
    #     "Marowak": "嘎啦嘎啦",
    #     "Nidoking": "尼多王",
    # }

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = torchvision.datasets.ImageFolder(root="./data/pokemon", transform=transform)
    train_dataset_size = int(0.7 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_dataset_size, len(dataset) - train_dataset_size])

    num_epochs = 10
    num_classes = len(dataset.classes)
    batch_size = 16
    lr = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -------------------- train -----------------
    train_loss_logs = []
    for epoch in range(1, num_epochs):
        model.train()
        cnt = 0
        train_one_loss = 0.0
        for iter_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # check loss
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
            train_one_loss += loss_value
            loss.backward()
            optimizer.step()

            if iter_idx % 7 == 0:
                print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, iter_idx + 1,
                                                                              len(train_loader), loss.item()))
        train_loss_logs.append((epoch, train_one_loss / len(train_loader)))

        # -------------------- evaluate -----------------
        if epoch % 5 == 0:
            accuracy = evaluate(model, val_loader)
            print('Accuracy: {:.2f}%'.format(accuracy))
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
            torch.save(model.state_dict(), 'checkpoint-{}-Acc-{:.2f}.pth'.format(epoch, accuracy))

        epochs = [sample[0] for sample in train_loss_logs]
        train_loss = [sample[1] for sample in train_loss_logs]
        plt.plot(epochs, train_loss, label='training loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Training loss curve')
