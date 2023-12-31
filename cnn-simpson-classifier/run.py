import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = ResidualBlock(3, 8, 2)
        self.layer2 = ResidualBlock(8, 16, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=4096, out_features=128)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=128, out_features=14)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

total_train_ds = torchvision.datasets.ImageFolder(
    root="/content/simpsons_dataset",
    transform=transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    ),
)

total_train_len = len(total_train_ds)
train_ds_len = int(total_train_len * 0.8)
valid_ds_len = total_train_len - train_ds_len

train_ds, valid_ds = random_split(total_train_ds, [train_ds_len, valid_ds_len])
test_ds = torchvision.datasets.ImageFolder(
    root="/content/simpsons_testset",
    transform=transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    ),
)

batch_size = 128
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size)
test_dl = DataLoader(test_ds, batch_size=batch_size)

img = next(iter(train_dl))
print(img[0].size())

print(len(train_ds))
print(len(valid_ds))
print(len(test_ds))

image, label = next(iter(train_dl))
plt.imshow(image[0].permute(1, 2, 0))
plt.show()
print(label)

model = CNN()
model.to(device)

criterion = nn.CrossEntropyLoss()
leraning_rate = 0.001

optimizer = torch.optim.Adam(model.parameters(), lr=leraning_rate)
print(model)

max_epoch = 10
train_loss_hist = []  # Train loss history
valid_loss_hist = []
valid_acc_hist = []  # Validation accuracy history


def train_one_epoch(data_loader):
    model.train()
    running_loss = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.detach()

    loss = (running_loss / len(data_loader)).item()
    return loss


# Validate the model
def eval(data_loader):
    model.eval()
    running_loss = 0

    with torch.no_grad():
        total = 0
        correct = 0

        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            # take the indices of the maximum values
            preds = torch.max(outputs, 1)[1]
            correct += (preds == labels).sum()
            total += len(labels)
            running_loss += loss.detach()

        loss = (running_loss / len(data_loader)).item()
        accuracy = (correct * 100 / total).item()
        return loss, accuracy


patience = 3
early_stopping_counter = 0
best_loss = float("inf")

# Training and validation loop
for epoch in range(1, max_epoch + 1):
    train_loss = train_one_epoch(train_dl)
    train_loss_hist.append(train_loss)

    # Evaluation
    valid_loss, valid_acc = eval(valid_dl)
    valid_loss_hist.append(valid_loss)
    valid_acc_hist.append(valid_acc)

    print(
        f"Epoch: {epoch}, Train Loss: {train_loss_hist[-1]}, Valid Loss: {valid_loss_hist[-1]}, Valid Accuracy: {valid_acc_hist[-1]}"
    )

    if valid_loss < best_loss:
        best_loss = valid_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), "/content/simpson_model_ckpt.pth")
        torch.save(optimizer.state_dict(), "/content/simpson_optim_ckpt.pth")
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print("Early Stopping!")
            break

# Visualize the training/validation loss and validation accuracy
epochs = list(range(1, max_epoch + 1))
plt.plot(epochs, train_loss_hist, label="Train Loss")
plt.plot(epochs, valid_loss_hist, label="Valid Loss")
plt.xlabel("#Epochs")
plt.ylabel("Loss")
plt.xticks(epochs)
plt.legend()
plt.show()

plt.plot(epochs, valid_acc_hist)
plt.xlabel("#Epochs")
plt.ylabel("Valid Accuracy")
plt.xticks(epochs)
plt.show()

# Load checkpoint
model = CNN().to(device)
model.load_state_dict(torch.load("/content/simpson_model_ckpt.pth"))

# Calculate test accuracy
test_loss, test_acc = eval(test_dl)
print(test_loss, test_acc)

# Accuracies per class
class_correct = [0.0 for _ in range(14)]
total_correct = [0.0 for _ in range(14)]

with torch.no_grad():
    for images, labels in test_dl:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.max(outputs, 1)[1]
        correct = (preds == labels).squeeze()

        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += correct[i].item()
            total_correct[label] += 1

idx_to_class = {v: k for k, v in total_train_ds.class_to_idx.items()}

for i in range(14):
    class_name = idx_to_class[i]
    print(f"class {class_name} accuracy: {class_correct[i] * 100 / total_correct[i]:.2f}%")
