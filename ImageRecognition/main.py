import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch import multiprocessing as mp




#Шаг 1. Загружаем датасет CIFAR-10
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset=torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader=torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset=torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader=torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes_set=('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#Шаг 2. Определение модели
class Net(nn.Module):
    def __init__(self):
        super (Net, self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6, 16, 5)
        self.fc1=nn.Linear(400, 120)
        self.fc2=nn.Linear(120, 84)
        self.fc3=nn.Linear(84, 10)

    def forward(self, x):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=x.view(-1, 400)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x

net=Net()

#Шаг 3. Определение функции потерь
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#Шаг 4. Обучение модели
if __name__ == '__main__':
    mp.freeze_support()
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
    
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Тренировка завершена')

#Шаг 5. Проверка точности модели
correct=0
total=0
with torch.no_grad():
    if __name__ == '__main__':
        mp.freeze_support()
        for data in testloader:
            images, labels=data
            outputs=net(images)
            _, predicted = torch.max(outputs.data, 1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()

        print("Точность сети на 10000 тестовых изображениях: %d %%" % (100 * correct / total))
