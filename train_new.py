"""
设置训练早停
"""
import argparse
import os
import time
from operator import truediv

import numpy as np
import hdf5storage
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from einops import rearrange
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from network import discriminator_ViT
from network import generator
from conloss import MultiSimilarityLoss

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def loadData():
    # 读入数据
    data = hdf5storage.loadmat('2015_122034/122034.mat')['ori_data']
    labels = hdf5storage.loadmat('2015_122034/122034_label3.mat')['map']

    return data, labels


# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX


# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1

    return patchesData, patchesLabels


def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                        y,
                                                        test_size=testRatio,
                                                        random_state=randomState,
                                                        stratify=y)

    return X_train, X_test, Y_train, Y_test


def create_data_loader():
    # 读入数据
    X, y = loadData()
    # 用于测试样本的比例
    test_ratio = args.test_ratio
    # 每个像素周围提取 patch 的尺寸
    patch_size = args.patch_size
    # 影像的波段数（波段维数）
    bands = args.bands

    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    X_pca = X

    print('\n... ... create data cubes ... ...')
    X_pca, Y_all = createImageCubes(X_pca, y, windowSize=patch_size)
    del X
    print('Data cube X shape: ', X_pca.shape)
    print('Data cube y shape: ', Y_all.shape)

    print('\n... ... create train & test data ... ...')
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, Y_all, test_ratio)
    
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtest  shape: ', Xtest.shape)

    # 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
    X = X_pca.reshape(-1, patch_size, patch_size, bands, 1)
    del X_pca
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, bands, 1)
    Xtest = Xtest.reshape(-1, patch_size, patch_size, bands, 1)
    print('before transpose: Xtrain shape: ', Xtrain.shape)
    print('before transpose: Xtest  shape: ', Xtest.shape)

    # 为了适应 pytorch 结构，数据要做 transpose
    X = X.transpose(0, 4, 3, 1, 2)
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    Xtest = Xtest.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)

    # 创建train_loader和 test_loader
    X = TestDS(X, Y_all)
    del Y_all
    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)
    del Xtrain,ytrain,Xtest,ytest
    Train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               )
    Test_loader = torch.utils.data.DataLoader(dataset=testset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=0,
                                              )
    All_data_loader = torch.utils.data.DataLoader(dataset=X,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  )

    return Train_loader, Test_loader, All_data_loader, y


""" Training dataset"""
class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


""" Testing dataset"""
class TestDS(torch.utils.data.Dataset):

    def __init__(self, Xtest, ytest):
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


def evaluate(net, Test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ps = []
    ys = []
    criterion = nn.CrossEntropyLoss()
    val_loss = 0
    for i, (x1, y1) in enumerate(Test_loader):
        with torch.no_grad():
            x1, y1 = x1.to(device), y1.to(device)
            p1 = net(x1)
            val_loss += criterion(p1, y1)

            p1 = p1.argmax(dim=1)
            ps.append(p1.detach().cpu().numpy())
            ys.append(y1.cpu().numpy())
    ps = np.concatenate(ps)
    ys = np.concatenate(ys)
    acc = np.mean(ys == ps)

    return acc, val_loss


def train(Train_loader, Test_loader, epochs):
    g = torch.Generator()
    g.manual_seed(233)
    # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 网络放到GPU上
    G_net = generator.Generator(n=64, imdim=27, imsize=[5, 5], zdim=10, device=0).to(device)
    G_optimizer = optim.Adam(G_net.parameters(), lr=args.lr)
    con_criterion = MultiSimilarityLoss(device=0)

    net = discriminator_ViT.ViT(num_classes=args.num_class).to(device)
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 初始化优化器
    D_optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # 变量
    best_acc = 0
    counter = 0
    patience = 10
    b_val_loss = 1000000
    D_LOSS_plt = []
    G_LOSS_plt = []
    Val_acc_plt = []
    Val_loss_plt = []

    # 开始训练
    for epoch in range(epochs):
        net.train()
        loop = tqdm(enumerate(Train_loader), total=len(Train_loader),
                    unit="pixel")  # total：迭代对象的总长度，用于计算进度条的百分比。如果不指定total，则默认为len(iterable)。
        right = 0
        D_LOSS = 0
        G_LOSS = 0
        loss_list = []

        for i, (data, target) in loop:
            data, target = data.to(device), target.to(device)
            y = target

            p_SD, z_SD = net(data, mode='train')
            D_loss1 = criterion(p_SD, y.long())

            data = rearrange(data, 'b c h w y -> b (c h) w y')
            with torch.no_grad():
                x_ED = G_net(data)
            rand = torch.nn.init.uniform_(torch.empty(len(data), 1, 1, 1)).to(device)  # Uniform distribution

            x_ID = rand * data + (1 - rand) * x_ED

            x_ED = rearrange(x_ED, 'b c h w -> b 1 c h w')
            x_ID = rearrange(x_ID, 'b c h w -> b 1 c h w')

            p_ED, z_ED = net(x_ED, mode='train')
            p_ID, z_ID = net(x_ID, mode='train')

            D_loss2 = criterion(p_ED, y.long()) + criterion(p_ID, y.long())

            # 判别器总损失
            D_loss = 0.4*D_loss1 + D_loss2
            # D_loss = D_loss1 + 0.2 * D_loss2
            D_optimizer.zero_grad()
            D_loss.backward(retain_graph=True)

            # 训练生成器

            # data = rearrange(data, 'b c h w y -> b (c h) w y')
            x_tgt = G_net(data)

            x_tgt = rearrange(x_tgt, 'b c h w -> b 1 c h w')
            p_tgt, z_tgt = net(x_tgt, mode='train')

            zsrc = torch.cat([z_SD.unsqueeze(1), z_tgt.unsqueeze(1)], dim=1)
            G_loss = con_criterion(zsrc, y)

            G_optimizer.zero_grad()
            G_loss.backward()

            D_optimizer.step()
            G_optimizer.step()

            _, predicted = torch.max(p_SD.data, 1)
            # 累加识别正确的样本数
            right += (predicted == y).sum()

            loss_list.append([D_loss.item(), G_loss.item(), D_loss1.item(), D_loss2.item()])
            D_LOSS, G_LOSS, D_loss1, D_loss2 = np.mean(loss_list, 0)

            loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')
            loop.set_postfix(Acc=float(right) / float(args.batch_size * i + len(data)), D_LOSS=D_LOSS, G_LOSS=G_LOSS, D_loss1=D_loss1, D_loss2=D_loss2)



        D_LOSS_plt.append(D_LOSS)
        G_LOSS_plt.append(G_LOSS)

        teacc, val_loss = evaluate(net, Test_loader)
        Val_acc_plt.append(teacc)
        Val_loss_plt.append(val_loss.item())

        if teacc > best_acc:
            best_acc = teacc
            torch.save(net.state_dict(), 'IEEE_TIP_SDEnet-main_1/results/2015model/122034.pkl')

        if val_loss < b_val_loss:
            b_val_loss = val_loss
            torch.save(net.state_dict(), 'IEEE_TIP_SDEnet-main_1/results/2015model/loss_122034.pkl')

        # # 判断是否早停
        # if teacc > best_acc or val_loss < b_val_loss:
        #     if teacc > best_acc:
        #         best_acc = teacc
        #         torch.save(net.state_dict(), './model/KNet.pkl')
        #
        #     if val_loss < b_val_loss:
        #         b_val_loss = val_loss
        #         torch.save(net.state_dict(), './model/loss_KNet.pkl')
        #     counter = 0
        # else:
        #     counter += 1
        #
        # # 判断是否触发早停
        # if counter >= patience:
        #     print("Early stopping!")
        #     break

    print('Finished Training')

    return D_LOSS_plt, G_LOSS_plt, Val_acc_plt, Val_loss_plt


def test(Test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 网络放到GPU上
    net = discriminator_ViT.ViT(num_classes=args.num_class).to(device)
    net.load_state_dict(torch.load('IEEE_TIP_SDEnet-main_1/results/2015model/122034.pkl'))

    count = 0
    # 模型测试
    net.eval()
    Y_pred_test = 0
    Y_test = 0

    for inputs, labels in Test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            Y_pred_test = outputs
            Y_test = labels
            count = 1
        else:
            Y_pred_test = np.concatenate((Y_pred_test, outputs))
            Y_test = np.concatenate((Y_test, labels))

    return Y_pred_test, Y_test


def AA_andEachClassAccuracy(Confusion_matrix):
    list_diag = np.diag(Confusion_matrix)
    list_raw_sum = np.sum(Confusion_matrix, axis=1)
    Each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(Each_acc)
    return Each_acc, average_acc


def acc_reports(Y_test, Y_pred_test):
    target_names = ['gengdi', 'lindi', 'jianzhuyongdi', 'caodi', 'shuiti']

    Classification = classification_report(Y_test, Y_pred_test, digits=4, target_names=target_names)
    OA = accuracy_score(Y_test, Y_pred_test)
    Confusion = confusion_matrix(Y_test, Y_pred_test)
    Each_acc, AA = AA_andEachClassAccuracy(Confusion)
    Kappa = cohen_kappa_score(Y_test, Y_pred_test)

    return Classification, OA * 100, Confusion, Each_acc * 100, AA * 100, Kappa * 100


def save_loss_to_file(loss_values, File_name):
    np.savetxt(File_name, loss_values, delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512, help='批量大小')
    parser.add_argument('--patch_size', type=int, default=5, help='图像块大小')
    parser.add_argument('--bands', type=int, default=27, help='波段数')
    parser.add_argument('--test_ratio', type=float, default=0.3, help='测试样本的比例')
    parser.add_argument('--num_class', type=int, default=3, help='类别数')
    parser.add_argument('--epochs', type=int, default=1, help='训练轮次')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    args = parser.parse_args()

    #  数据划分
    train_loader, test_loader, all_data_loader, y_all = create_data_loader()

    #  训练网络
    tic1 = time.perf_counter()
    D_loss_plt, G_loss_plt, val_acc_plt, val_loss_plt = train(train_loader, test_loader, epochs=args.epochs)
    toc1 = time.perf_counter()

    #  验证集输入
    tic2 = time.perf_counter()
    y_pred_test, y_test = test(test_loader)
    toc2 = time.perf_counter()

    # 验证集评价指标
    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
    classification = str(classification)

    Training_Time = toc1 - tic1
    Test_time = toc2 - tic2


    #save_loss_to_file(D_loss_plt, 'result/D_loss.csv')
    #save_loss_to_file(G_loss_plt, 'result/G_loss.csv')
    #save_loss_to_file(val_acc_plt, 'result/val_acc.csv')
    #save_loss_to_file(val_loss_plt, 'result/val_loss.csv')

    # 绘制折线图
    #plt.plot(D_loss_plt, label='D_LOSS')
    #plt.plot(G_loss_plt, label='G_LOSS')
    #plt.plot(val_acc_plt, label='Validation Accuracy')
    #plt.plot(val_loss_plt, label='Validation Loss')

    # 添加图例
    #plt.legend()

    # 显示图形
    #plt.show()

    file_name = "IEEE_TIP_SDEnet-main_1/results/classification_report.txt"
    with open(file_name, 'w') as x_file:
        x_file.write('{} Training_Time (s)'.format(Training_Time))
        x_file.write('\n')
        x_file.write('{} Test_time (s)'.format(Test_time))
        x_file.write('\n')
        x_file.write('{} Kappa accuracy (%)'.format(kappa))
        x_file.write('\n')
        x_file.write('{} Overall accuracy (%)'.format(oa))
        x_file.write('\n')
        x_file.write('{} Average accuracy (%)'.format(aa))
        x_file.write('\n')
        x_file.write('{} Each accuracy (%)'.format(each_acc))
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion))

