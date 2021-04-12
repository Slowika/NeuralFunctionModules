import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.preresnet import PreActResNet, PreActBlock
from models.attentive_densenet import AttentiveDensenet

class ConvInputModel_NFL(nn.Module):
    def __init__(self):
        super(ConvInputModel_NFL, self).__init__()

        self.initial_channels = 24

        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 48, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(48, 96, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(96)
        self.conv4 = nn.Conv2d(96, 192, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(192)

        layer_channels = [self.initial_channels] + [self.initial_channels*2] + [self.initial_channels*4] + [self.initial_channels*8]

        self.ad = AttentiveDensenet(layer_channels, key_size=16, val_size=16, n_heads=4, att_sparsity=3)

    def forward(self, img):
        self.ad.reset()

        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)

        x = self.ad(x, read=True, write=True)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)

        x = self.ad(x, read=True, write=True)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)

        x = self.ad(x, read=True, write=True)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)

        x = self.ad(x, read=True, write=True)

        x = F.avg_pool2d(x, 4)
        x = F.interpolate(x, [5, 5])

        return x


# CNN-NFL
class ConvInputModel_NFL2(nn.Module):
    def __init__(self):
        super(ConvInputModel_NFL2, self).__init__()
        self.initial_channels = 24*2
        self.conv1 = nn.Conv2d(3, 24*2, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24*2)
        self.conv2 = nn.Conv2d(24*2, 48*2, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(48*2)
        self.conv3 = nn.Conv2d(48*2, 96*2, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(2*96)
        self.conv4 = nn.Conv2d(2*96, 2*192, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(2*192)

        self.conv1b = nn.Conv2d(3, 2*24, 3, stride=2, padding=1)
        self.batchNorm1b = nn.BatchNorm2d(2*24)
        self.conv2b = nn.Conv2d(2*24, 2*48, 3, stride=2, padding=1)
        self.batchNorm2b = nn.BatchNorm2d(2*48)
        self.conv3b = nn.Conv2d(2*48, 2*96, 3, stride=2, padding=1)
        self.batchNorm3b = nn.BatchNorm2d(2*96)
        self.conv4b = nn.Conv2d(2*96, 2*192, 3, stride=2, padding=1)
        self.batchNorm4b = nn.BatchNorm2d(2*192)
        layer_channels = [self.initial_channels] + [self.initial_channels * 2] + [self.initial_channels * 4] + [
        self.initial_channels * 8] + [self.initial_channels] + [self.initial_channels * 2] + [
                         self.initial_channels * 4] + [self.initial_channels * 8]
        self.ad = AttentiveDensenet(layer_channels, key_size=16, val_size=16, n_heads=4, att_sparsity=5)


    def forward(self, img):
        self.ad.reset()
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.ad(x, read=False, write=True)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.ad(x, read=False, write=True)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.ad(x, read=False, write=True)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        x = self.ad(x, read=False, write=True)

        x = self.conv1b(img)
        x = F.relu(x)
        x = self.batchNorm1b(x)
        x = self.ad(x, read=True, write=True)
        x = self.conv2b(x)
        x = F.relu(x)
        x = self.batchNorm2b(x)
        x = self.ad(x, read=True, write=True)
        x = self.conv3b(x)
        x = F.relu(x)
        x = self.batchNorm3b(x)
        x = self.ad(x, read=True, write=True)
        x = self.conv4b(x)
        x = F.relu(x)
        x = self.batchNorm4b(x)
        x = self.ad(x, read=True, write=True)
        x = F.avg_pool2d(x, 4)
        x = F.interpolate(x, [5, 5])
        return x

class FCOutputModel(nn.Module):
    def __init__(self):
        super(FCOutputModel, self).__init__()

        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x)


class BasicModel(nn.Module):
    def __init__(self, args, name):
        super(BasicModel, self).__init__()
        self.name = name
        self.args = args

    def train_(self, input_img, input_qst, label):
        self.optimizer.zero_grad()
        output = self(input_img, input_qst)
        loss = F.nll_loss(output, label)
        loss.backward()
        self.optimizer.step()
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return (accuracy, loss)
        
    def test_(self, input_img, input_qst, label):
        output = self(input_img, input_qst)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy

    def save_model(self, epoch):
        torch.save(self.state_dict(), self.args.experiment_name + '/epoch_{}_{:02d}.pth'.format(self.name, epoch))


class Preresnet_MLP(BasicModel):
    def __init__(self, args):
        super(Preresnet_MLP, self).__init__(args, 'PreresnetMLP')

        self.conv = PreActResNet(PreActBlock, [2,2,2,2], initial_channels=24, num_classes=10,  per_img_std=False, stride= 1, use_attentive_densenet=True, ad_heads=4,att_sparsity=None)
        self.conv1 = nn.Conv2d(192*2, 24, 1, stride=1, padding=0)
        self.fc1 = nn.Linear(5 * 5 * 24 + 11, 256)
        self.fcout = FCOutputModel()

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

    def forward(self, img, qst):
        x = self.conv(img)
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x_ = torch.cat((x, qst), 1)
        x_ = self.fc1(x_)
        x_ = F.relu(x_)

        return self.fcout(x_)


class CNN_MLP_NFL(BasicModel):
    def __init__(self, args):
        super(CNN_MLP_NFL, self).__init__(args, 'CNNMLPNFL')

        self.conv = ConvInputModel_NFL2()
        self.conv1 = nn.Conv2d(192*2, 64, 1, stride=1, padding=0)

        self.fc1 = nn.Linear(5 * 5 * 64 + 11, 256)
        self.fcout = FCOutputModel()

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

    def forward(self, img, qst):
        x = self.conv(img)
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x_ = torch.cat((x, qst), 1)
        x_ = self.fc1(x_)
        x_ = F.relu(x_)
        #x_ = F.dropout(x_)

        return self.fcout(x_)

