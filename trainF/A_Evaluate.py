# 计算正确率
import torch

from trainF.Out_logits import out_logits


def evaluate(model, loader, criterion, center_loss, device, alpha):
    """
    :param model: 网络模型
    :param loader: 数据集
    :return: 正确率
    """
    correct = 0
    loss_sum = 0
    total = len(loader.dataset)
    for x1, x2, x3,x4, y in loader:
        with torch.no_grad():
            logits1 = model(x1, x2, x3,x4).to(device)
            loss1 = criterion(logits1, y.to(device))
            loss2 = center_loss(logits1,y.to(device))
            loss = alpha*loss1 + (1-alpha)*loss2
            pred = logits1.argmax(dim=1)
        correct += torch.eq(pred, y.to(device)).sum().float().item()
        acc = correct / total
        loss_sum += loss
        loss_back = loss_sum / total
    return acc, loss_back


def evaluate_df(model, loader, criterion, device, alpha):
    """
    :param model: 网络模型
    :param loader: 数据集
    :return: 正确率
    """
    correct = 0
    loss_sum = 0
    total = len(loader.dataset)
    for x1, x2, x3,x4, y in loader:
        with torch.no_grad():
            logits1 = model(x1).to(device)
            loss1 = criterion(logits1, y.to(device))
            # loss2 = center_loss(logits1,y.to(device))
            loss = loss1
            pred = logits1.argmax(dim=1)
        correct += torch.eq(pred, y.to(device)).sum().float().item()
        acc = correct / total
        loss_sum += loss
        loss_back = loss_sum / total
    return acc, loss_back