import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sklearn.metrics as m

from trainF.Out_logits import out_logits

# 计算测试准确率
def test_evaluate(model, loader, version, device):
    """
       :param model: 网络模型
       :param loader: 数据集
       :return: 正确率
    """
    y_true = []
    predict = []
    predict_pro = []
    for x1, x2, x3,x4, y in loader:
        with torch.no_grad():
            logits = model(x1, x2, x3,x4).to(device)
            result = logits.argmax(dim=1)
            prob = (logits.T)[1]
            for i in y.numpy():
                y_true.append(i)
            for j in result.cpu().numpy():
                predict.append(j)
            for k in prob.cpu().numpy():
                predict_pro.append(k)

    print(classification_report(y_true, predict, digits=4))
    # plot_confusion_matrix(confusion_matrix(y_true, predict), classes=range(2), normalize=True, title='confusion matrix')
    print("混淆矩阵")
    print(confusion_matrix(y_true, predict))
    print("f1-score:{}.".format(m.f1_score(y_true, predict)))
    # plot_roc(y_true, predict_pro)
    return accuracy_score(y_true, predict), confusion_matrix(y_true, predict)


# 计算测试准确率
def test_evaluate_df(model, loader, version, device):
    """
       :param model: 网络模型
       :param loader: 数据集
       :return: 正确率
    """
    y_true = []
    predict = []
    predict_pro = []
    for x1, x2, x3,x4, y in loader:
        with torch.no_grad():
            logits = model(x1).to(device)
            result = logits.argmax(dim=1)
            prob = (logits.T)[1]
            for i in y.numpy():
                y_true.append(i)
            for j in result.cpu().numpy():
                predict.append(j)
            for k in prob.cpu().numpy():
                predict_pro.append(k)

    print(classification_report(y_true, predict, digits=4))
    # plot_confusion_matrix(confusion_matrix(y_true, predict), classes=range(2), normalize=True, title='confusion matrix')
    print("混淆矩阵")
    print(confusion_matrix(y_true, predict))
    print("f1-score:{}.".format(m.f1_score(y_true, predict)))
    # plot_roc(y_true, predict_pro)
    return accuracy_score(y_true, predict), confusion_matrix(y_true, predict)