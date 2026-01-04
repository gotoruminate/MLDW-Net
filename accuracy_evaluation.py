"""
accuracy_evaluation.py
独立精度计算脚本，支持TIFF格式的预测结果

使用方法：
1. 准备真实标签的.mat文件和预测结果的.tif文件
2. 修改下面的文件路径配置
3. 运行本脚本

输出：
- 控制台打印精度报告
- 保存精度报告到文本文件
"""

import numpy as np
import hdf5storage
import tifffile as tiff
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import argparse

def AA_andEachClassAccuracy(confusion_matrix):
    """
    计算每个类别的精度和平均精度
    """
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def load_data(label_path, prediction_path):
    """
    加载标签和预测结果数据
    """
    # 加载真实标签（.mat格式）
    labels = hdf5storage.loadmat(label_path)['map']
    
    # 加载预测结果（.tif格式）
    predictions = tiff.imread(prediction_path)
    
    return labels, predictions

def preprocess_data(labels, predictions):
    """
    预处理数据，准备用于精度计算
    """
    # 展平数组
    labels_flat = labels.flatten()
    predictions_flat = predictions.flatten()
    
    # 只保留有效标签（非零）
    valid_mask = labels_flat > 0
    y_true = labels_flat[valid_mask] - 1  # 减去1以匹配训练时的标签处理方式
    y_pred = predictions_flat[valid_mask] - 1  # 预测结果也是1-based，需要转换为0-based
    
    return y_true, y_pred

def calculate_metrics(y_true, y_pred, class_names):
    """
    计算所有精度指标
    """
    # 计算各项指标
    classification_rep = classification_report(y_true, y_pred, digits=4, target_names=class_names)
    oa = accuracy_score(y_true, y_pred)
    confusion_mat = confusion_matrix(y_true, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion_mat)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # 返回结果字典
    return {
        'classification_report': classification_rep,
        'overall_accuracy': oa * 100,
        'average_accuracy': aa * 100,
        'kappa_coefficient': kappa * 100,
        'confusion_matrix': confusion_mat,
        'each_class_accuracy': each_acc * 100
    }

def save_results(metrics, output_path):
    """
    保存精度结果到文件
    """
    with open(output_path, 'w') as f:
        f.write("Classification Report:\n")
        f.write(metrics['classification_report'])
        f.write("\n\nOverall Accuracy: {:.2f}%\n".format(metrics['overall_accuracy']))
        f.write("Average Accuracy: {:.2f}%\n".format(metrics['average_accuracy']))
        f.write("Kappa Coefficient: {:.2f}%\n".format(metrics['kappa_coefficient']))
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(metrics['confusion_matrix']))
        f.write("\n\nEach Class Accuracy:\n")
        f.write(np.array2string(metrics['each_class_accuracy']))

def main():
    # 设置参数解析器
    parser = argparse.ArgumentParser(description='分类模型精度评估（支持TIFF格式预测结果）')
    parser.add_argument('--label_path', type=str, default='2005_121033/121033_label2.mat',
                       help='真实标签.mat文件路径')
    parser.add_argument('--prediction_path', type=str, default='IEEE_TIP_SDEnet-main_1/results/121033.tif',
                       help='预测结果.tif文件路径')
    args = parser.parse_args()

    # 定义类别名称（根据你的实际类别修改）
    class_names = ['qitayongdi', 'lindi', 'jianzhuyongdi', 'shuiti', 'xiaomai']

    print("\n开始精度计算...")
    print(f"标签文件: {args.label_path}")
    print(f"预测文件: {args.prediction_path}")

    # 1. 加载数据
    labels, predictions = load_data(args.label_path, args.prediction_path)
    print("\n数据加载完成:")
    print(f"标签数据形状: {labels.shape}")
    print(f"预测数据形状: {predictions.shape}")

    # 2. 预处理数据
    y_true, y_pred = preprocess_data(labels, predictions)
    print(f"\n有效样本数: {len(y_true)}")

    # 检查预测结果的有效性
    unique_pred = np.unique(y_pred)
    print(f"预测结果中的唯一值: {unique_pred}")
    if not all(0 <= val < len(class_names) for val in unique_pred):
        print("警告: 预测结果包含超出类别范围的数值！")

    # 3. 计算精度指标
    metrics = calculate_metrics(y_true, y_pred, class_names)

    # 4. 打印结果
    print("\n\n分类性能报告:")
    print("="*50)
    print(metrics['classification_report'])
    print("\n总体精度 (OA): {:.2f}%".format(metrics['overall_accuracy']))
    print("平均精度 (AA): {:.2f}%".format(metrics['average_accuracy']))
    print("Kappa系数: {:.2f}%".format(metrics['kappa_coefficient']))
    print("\n混淆矩阵:")
    print(metrics['confusion_matrix'])
    print("\n各类别精度:")
    for name, acc in zip(class_names, metrics['each_class_accuracy']):
        print(f"{name}: {acc:.2f}%")


if __name__ == '__main__':
    main()