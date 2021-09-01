import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
from tqdm import tqdm
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

train_dataset = datasets.ImageFolder("data/train", train_transforms)
test_dataset = datasets.ImageFolder("data/test", test_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=12, shuffle=True, num_workers=4)

model = models.resnet18(pretrained=True)
set_parameter_requires_grad(model, True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)

optimizer = torch.optim.SGD(model.parameters(), lr=0.00002)
loss_func = torch.nn.MSELoss()

if torch.cuda.is_available():
    model.to(device)

if os.path.exists("model.dat"):
    model.load_state_dict(torch.load("model.dat"))

prev_test_loss = 1000
thresh = 0.0001
for t in tqdm(range(200)):
    for train_features, train_labels in tqdm(train_loader):
        train_features = train_features.to(device)
        train_labels = train_labels.to(device)

        prediction = model(train_features).squeeze()
        loss = loss_func(prediction.float(), train_labels.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    total_loss = 0
    for test_features, test_labels in tqdm(test_loader):
        test_features = test_features.to(device)
        test_labels = test_labels.to(device)

        prediction = model(test_features).squeeze()
        loss = loss_func(prediction.float(), test_labels.float())

        total_loss += float(loss)

    avg_test_loss = total_loss / len(test_loader)
    print("avg test loss={loss}".format(loss=avg_test_loss))

    torch.save(model.state_dict(), "model.dat")

    if abs(avg_test_loss - prev_test_loss) <= thresh:
        break

    prev_test_loss = avg_test_loss