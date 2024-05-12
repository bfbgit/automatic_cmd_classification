import torch
from torch.utils.data import DataLoader
from loadECG.DataSet import ECGDataSet
from net.A_Fusion import Fusion_Model
from trainF.A_Test_evaluate import test_evaluate
import time



def main():
    batchsz = 50
    lr = 1e-3  # 学习率
    torch.manual_seed(1234)

    model = Fusion_Model()

    filename = r'DATA_Test01.pkl'
    version = 'v_DATA_Fusion_model'
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


    # 读取数据
    test_db = ECGDataSet(filename, mode='test')
    print(test_db)

    test_loader = DataLoader(test_db, batch_size=batchsz, shuffle=True)
    print(test_loader)

    print("Data loaded successfully!")

    model_save_pth = './models/{}'.format(version)
    model.load_state_dict(torch.load(model_save_pth))
    print("loaded from ckpt!")

    test_acc, cm = test_evaluate(model, test_loader, version, device)
    print("test_acc:{}".format(test_acc))


if __name__ == '__main__':
    
    main()
    
