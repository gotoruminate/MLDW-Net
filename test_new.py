import os
import numpy as np
import hdf5storage
import tifffile as tiff
import torch
from network import discriminator_ViT
import argparse
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from matplotlib import pyplot as plt

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:32"
cudnn.enabled = True



def loadData_A():
    # 读入数据
    data = hdf5storage.loadmat('2015_data/2015_122034_save.mat')['ori_data']
    # data = sio.loadmat(r'D:\PY\最终代码各类样本相同\data\chuanhuiqu.mat')['chuanhuiqu']
    return data


# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX


def createImageCubes_A(X, windowSize=5, removeZeroLabels=True):
    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchIndex = patchIndex + 1

    return patchesData


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', type=int, default=3, help='类别数')
    parser.add_argument('--patch_size', type=int, default=5, help='图像块大小')
    parser.add_argument('--bands', type=int, default=27, help='波段数')
    parser.add_argument('--in_size', type=int, default=5, help='多行输入')
    parser.add_argument('--batch_size', type=int, default=4096, help='推理批大小')
    args = parser.parse_args()

    # 读入数据
    data_a = loadData_A()

    # 每个像素周围提取 patch 的尺寸
    patch_size = args.patch_size

    # 影像的波段数（波段维数）
    pca_components = args.bands

    row = data_a.shape[0]
    col = data_a.shape[1]
    print(row, col)
    out = np.zeros(row * col)

    in_size = args.in_size
    X_data = data_a

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        net = discriminator_ViT.ViT(num_classes=args.num_class).to(device)
        net.load_state_dict(torch.load('IEEE_TIP_SDEnet-main_1/results/2015model/122034.pkl'))
        net.eval()

    # 网络放到GPU上
    #net = discriminator_ViT.ViT(num_classes=args.num_class).to(device)
    #net.load_state_dict(torch.load('IEEE_TIP_SDEnet-main_1/results/KNet.pkl'))
    # net.load_state_dict(torch.load('model/loss_KNet.pkl'))
    #net.eval()

    for i in tqdm(range(row // in_size), desc="Processing", unit="piece"):
        if i == 0:
            X = createImageCubes_A(X_data[i * in_size:(i + 1) * in_size + int((patch_size - 1) / 2), :, :],
                                   windowSize=patch_size)
            # print(X.shape)
            # (30926, 5, 5, 28)
            len_a = X.shape[0]
            X = X[:len_a - int((patch_size - 1) / 2) * col, :, :, :]
            X = X.reshape(-1, patch_size, patch_size, pca_components, 1)
            X = X.transpose(0, 4, 3, 1, 2)

        elif i == (row // in_size) - 1:
            X = createImageCubes_A(X_data[i * in_size - int((patch_size - 1) / 2):, :, :], windowSize=patch_size)
            # (1000*1000, 5, 5, 28)
            # len = X.shape[0]
            X = X[int((patch_size - 1) / 2) * col:, :, :, :]

            X = X.reshape(-1, patch_size, patch_size, pca_components, 1)
            X = X.transpose(0, 4, 3, 1, 2)

        else:
            X = createImageCubes_A(
                X_data[i * in_size - int((patch_size - 1) / 2):(i + 1) * in_size + int((patch_size - 1) / 2), :, :],
                windowSize=patch_size)
            # (1000*1000, 5, 5, 28)
            len_b = X.shape[0]
            X = X[int((patch_size - 1) / 2) * col:len_b - int((patch_size - 1) / 2) * col, :, :, :]

            X = X.reshape(-1, patch_size, patch_size, pca_components, 1)
            X = X.transpose(0, 4, 3, 1, 2)

        with torch.no_grad():
            # 计算当前分块的起始和结束索引（全局）
            if i == ((row // in_size) - 1):
                out_start = i * in_size * col
                out_end = row * col  # 最后一块可能不完整
            else:
                out_start = i * in_size * col
                out_end = (i + 1) * in_size * col

            # 分批处理
            batch_size = args.batch_size
            for batch_idx in range(0, X.shape[0], batch_size):
                batch_X = X[batch_idx : batch_idx + batch_size]
        
                inputs = torch.FloatTensor(batch_X).to(device)
                outputs = net(inputs)
        
                # 计算当前 batch 在 out 中的位置
                current_batch_size = outputs.shape[0]  # 实际 batch 大小（最后一批可能较小）
                out_batch_start = out_start + batch_idx
                out_batch_end = out_batch_start + current_batch_size
        
                # 确保不超过 out_end
                out_batch_end = min(out_batch_end, out_end)
        
                # 存储结果
                out[out_batch_start : out_batch_end] = np.argmax(
                    outputs.detach().cpu().numpy(), axis=1
                ).astype(int)
        
                del inputs, outputs
                torch.cuda.empty_cache()

        # print('**********', i)

    prediction = np.reshape(out, (row, col)) + 1

    plt.imshow(prediction)
    plt.show()
    tiff.imwrite('IEEE_TIP_SDEnet-main_1/results/2015result/122034.tif', prediction)
    # tiff.imwrite('./result/result_chuanhuiqu.tif', prediction)

