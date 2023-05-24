import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# 定义网络结构
class ActionNet(nn.Module):
    def __init__(self):
        super(ActionNet, self).__init__()
        self.fc1 = nn.Linear(56 * 768, 128) # 扁平化后的输入大小为 56*768
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)  # 四种动作，所以最后一层的输出大小为4

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 对输入张量进行扁平化处理
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        output = self.fc3(x)
        return output


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    iter_wrapper = (lambda x: tqdm(x, total=len(train_loader))) if False else (lambda x: x)
    for batch_idx, (data, target) in iter_wrapper(enumerate(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    torch.save(model, outfile+'classifier.pth')

def validate(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print('\nAccuracy on test set: {:.0f}%\n'.format(100. * correct / total))

def train_main(outfile):
    # 假设你的输入张量和目标张量是input和target
    data = torch.load(outfile +'atari.pt')
    print("Load %d data from split(s) %s." % (len(data[0]), 'atari'))
    input = data[0]
    target = data[3]

    # # 假设 `data` 是你的数据列表
    # X = [x[0] for x in data]
    # y = [x[1] for x in data]

    # 将数据分为训练集和测试集，这里我们使用了80%/20%的划分
    X_train, X_test, y_train, y_test = train_test_split(input, target, test_size=0.2, random_state=42)

    # 转换为PyTorch张量
    # X_train = torch.stack(X_train)
    y_train = y_train.clone().detach().long()
    # y_train = torch.tensor(y_train, dtype=torch.long)
    # X_test = torch.stack(X_test)
    y_test = y_test.clone().detach().long()
    # y_test = torch.tensor(y_test, dtype=torch.long)

    # 创建DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 2):
        train(model, device, train_loader, optimizer, criterion, epoch)
        validate(model, device, test_loader)
        

def test_main(outfile):
    model = torch.load(outfile + 'classifier.pth')
        # 假设你的输入张量和目标张量是input和target
    data = torch.load(outfile + 'atari.pt')
    print("Load %d data from split(s) %s." % (len(data[0]), 'atari'))
    input = data[0]
    target = data[3]

    # # 假设 `data` 是你的数据列表
    # X = [x[0] for x in data]
    # y = [x[1] for x in data]

    # 将数据分为训练集和测试集，这里我们使用了80%/20%的划分
    X_train, X_test, y_train, y_test = train_test_split(input, target, test_size=0.4, random_state=40)

    # 转换为PyTorch张量
    # X_train = torch.stack(X_train)
    y_train = y_train.clone().detach().long()
    # y_train = torch.tensor(y_train, dtype=torch.long)
    # X_test = torch.stack(X_test)
    y_test = y_test.clone().detach().long()
    # y_test = torch.tensor(y_test, dtype=torch.long)

    # 创建DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validate(model, device, test_loader)

if __name__ == '__main__':
    outfile = './data/atari_imgfeat/'
    test_main(outfile)


















