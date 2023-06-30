"""
特征提取的实例：
利用迁移学习中特征提取的方法来对CIFAR-10数据集实现对10类无体的分类
"""
from utils import *
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 64
epochs = 30

# 加载和预处理数据集
trans_train = transforms.Compose(
    [transforms.RandomResizedCrop(224),  # 将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为制定的大小；
     # （即先随机采集，然后对裁剪得到的图像缩放为同一大小） 默认scale=(0.08, 1.0)
     transforms.RandomHorizontalFlip(),  # 以给定的概率随机水平旋转给定的PIL的图像，默认为0.5；
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])

trans_valid = transforms.Compose(
    [transforms.Resize(256),  # 是按照比例把图像最小的一个边长放缩到256，另一边按照相同比例放缩。
     transforms.CenterCrop(224),  # 依据给定的size从中心裁剪
     transforms.ToTensor(),  # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])  # 对数据按通道进行标准化

train_set = torchvision.datasets.CIFAR10(root="./cifar10", train=True, download=False, transform=trans_train)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='./cifar10', train=False,
                                        download=False, transform=trans_valid)
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

if __name__ == '__main__':

    writer1 = SummaryWriter("./runs/loss")
    writer2 = SummaryWriter("./runs/acc")

    model = get_vit_model()

    # 查看总参数及训练参数
    total_params = sum(p.numel() for p in model.parameters())
    print('总参数个数:{}'.format(total_params))

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()  # 损失函数
    # 只需要优化最后一层参数
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-3, momentum=0.9)  # 优化器

    prev_time = datetime.now()

    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        net = model.train()#启用 BatchNormalization 和 Dropout
        for image, label in trainloader:
            image = image.to(device)
            label = label.to(device)
            # forward
            output = net(image)
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += get_acc(output, label)

            cur_time = datetime.now()

        writer1.add_scalar("loss", train_loss / len(trainloader), epoch)
        writer2.add_scalar("acc_train", train_acc / len(trainloader), epoch)

        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)

        with torch.no_grad():#梯度不在反向传播，减少计算
            valid_loss = 0
            valid_acc = 0
            net = net.eval()#不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化
            for image, label in testloader:
                image = image.to(device)
                label = label.to(device)
                output = net(image)
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)
            epoch_str = (
                    "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                    % (epoch, train_loss / len(trainloader),
                       train_acc / len(trainloader), valid_loss / len(testloader),
                       valid_acc / len(testloader)))

        prev_time = cur_time
        print(epoch_str + time_str)

    writer1.close()
    writer2.close()
