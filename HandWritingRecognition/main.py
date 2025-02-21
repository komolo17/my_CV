import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch import multiprocessing as mp
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
import numpy as np

#Гиперпараметры
epochs=6
classes=10
batch_size=100

#Загружаем датасет MNIST
transform=transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))
     ])

trainset=torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
trainloader=torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset=torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
testloader=torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

#Определяем модель
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1=nn.Linear(28*28,200)
        self.fc2=nn.Linear(200, 200)
        self.fc3=nn.Linear(200, 10)

    def forward(self, x):
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return torch.log_softmax(x, dim=1)

net=Net()
print(net)

#Осуществляем оптимизацию путем стохастического градиентного спуска
optimizer=optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#Создаем функцию потерь
criterion=nn.NLLLoss()

#Запускаем главный тренировочный цикл
loss_list=[]
acc_list=[]
if __name__=='__main__':
    mp.freeze_support()
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(trainloader):
           images=images.view(-1, 28*28)
           optimizer.zero_grad()
           net_out=net(images)
           loss=criterion(net_out, labels)
           loss_list.append(loss.item())
           loss.backward()
           optimizer.step()

           total = labels.size(0)
           _, predicted = torch.max(net_out.data, 1)
           correct = (predicted == labels).sum().item()
           acc_list.append(correct/total)
           if (i+1)%100==0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, epochs, i + 1, trainloader, loss.item(),
                              (correct / total) * 100))

#Тестируем модель
net.eval()
with torch.no_grad():
    correct=0
    total=0
    for images, labels in testloader:
        images = images.view(-1, 28 * 28)
        outputs=net(images)
        _, predicted=torch.max(outputs.data, 1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
    print('Точность модели на 10000 тестовых изображений: {} %'.format((correct/total)*100))

torch.save(net.state_dict(), './net' + 'conv_net_model.ckpt')

#Выводим графики
p=figure(y_axis_label='Потери', width=850, y_range=(0, 1), title="Результаты")
p.extra_y_ranges={'Точность': Range1d(start=0, end=100)}
p.add_layout(LinearAxis(y_range_name='Точность', axis_label='Точность(%)'), 'right')
p.line(np.arange(len(loss_list)), loss_list)
p.line(np.arange(len(loss_list)), np.array(acc_list)*100, y_range_name='Точность', color='red')
show(p)