import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import sklearn.metrics as m
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from net.A_Fusion import Fusion_Model
from loadECG.DataSet import ECGDataSet
from trainF.A_Evaluate import evaluate
from trainF.A_Test_evaluate import test_evaluate
from trainF.centerloss import Centerloss



def main():
    batchsz = 50
    lr = 1e-3  # 学习率
    epoches = 30
    torch.manual_seed(1234)


    model = Fusion_Model()

    filename = r'../DATA/DATA_0618.pkl'
    version = 'v_DATA_Fusion_model'
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # 读取数据
    train_db = ECGDataSet(filename, mode='train')
    test_db = ECGDataSet(filename, mode='test')

    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True)
    test_loader = DataLoader(test_db, batch_size=batchsz, shuffle=True)


    optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()  # 设置损失函数
    center_loss = Centerloss().to(device)

    log_dir = "./logs/{}".format(version)
    model_save_pth = './models/{}'.format(version)
    writer = SummaryWriter(log_dir=log_dir)

    best_epoch, best_acc = 0, 0
    alpha = 0.9

    for epoch in range(epoches):
        for step, (x1, x2, x3, x4, y) in enumerate(tqdm(train_loader)):
            # x1: torch.Size([batchsz, 12, 300])   x2:torch.Size([batchsz, 12, 11])
            logits1 = model(x1, x2, x3, x4).to(device)
            loss1 = criterion(logits1, y.to(device))
            loss2 = center_loss(logits1, y.to(device))
            loss = alpha*loss1 + (1-alpha)*loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 1 == 0:
            train_acc, train_loss = evaluate(model, train_loader,criterion, center_loss, device, alpha)
            test_acc, test_loss = evaluate(model, test_loader, criterion, center_loss, device, alpha)
            test_acc, cm = test_evaluate(model, test_loader, version, device)
            print("epoch:[{}/{}]. val_acc:{}. test_loss:{}.  train_acc:{}.  train_loss:{}".format(epoch + 1, epoches,
                                                                                                  test_acc, test_loss,
                                                                                                  train_acc,
                                                                                                  train_loss))

            writer.add_scalar('CMD_acc/train_acc', train_acc, epoch)
            writer.add_scalar('CMD_loss/train_loss', train_loss.item(), epoch)
            writer.add_scalar('CMD_acc/test_acc', test_acc, epoch)
            writer.add_scalar('CMD_loss/test_loss', test_loss.item(), epoch)
            if test_acc >= best_acc:
                best_epoch = epoch
                best_acc = test_acc
                torch.save(model.state_dict(), model_save_pth)

    print('best acc:{}. best epoch:{}.'.format(best_acc, best_epoch + 1))



if __name__ == '__main__':
    main()
