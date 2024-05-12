import torch
from torch.utils.data import Dataset
from loadECG.load_pkl import loadData_noSplit


class ECGDataSet(Dataset):
    """
    MyDataSet类继承自torch中的DataSet实现对数据集的读取
    """
    def __init__(self, root, mode):
        """
        :param root: 数据集的路径
        :param mode: train,val,test
        """
        super(ECGDataSet, self).__init__()

        self.mode = mode   # 设置读取读取数据集的模式
        self.root = root   # 数据集存放的路径
        label = []

        signal, tf, cfr_mce,imr, label = loadData_noSplit(filename=root)
        # with open(root, 'r') as f:    # 从csv中读取数据
        #     reader = csv.reader(f)
        #     result = list(reader)
        #     del result[0]             # 删除表头
        #     random.shuffle(result)
        #     for i in range(len(result)):
        #         del result[i][0]
        #         label.append(int(result[i][0]))
        #         del result[i][0]
        # result = np.array(result, dtype=np.float)
        # result = preprocessing.scale(result).tolist()
        # result = preprocessing.StandardScaler().fit_transform(result).tolist()  # 对数据进行预处理
        # result = preprocessing.MinMaxScaler().fit_transform(result).tolist()
        assert len(signal) == len(label)
        assert len(tf) == len(label)
        self.labels = label
        self.tfs = tf
        self.cfr_mce = cfr_mce
        self.imr=imr
        self.datas = signal
        
        if mode == 'train':  # 划分训练集为60%
            self.datas = self.datas[:int(0.8 * len(self.datas))]
            self.tfs = self.tfs[:int(0.8 * len(self.tfs))]
            self.cfr_mce = self.cfr_mce[:int(0.8 * len(self.cfr_mce))]
            self.imr = self.imr[:int(0.8 * len(self.imr))]
            self.labels = self.labels[:int(0.8 * len(self.labels))]
        else:                  # 划分测试集为20%
            self.datas = self.datas[int(0.8 * len(self.datas)):]
            self.tfs = self.tfs[int(0.8 * len(self.tfs)):]
            self.cfr_mce = self.cfr_mce[int(0.8 * len(self.cfr_mce)):]
            self.imr = self.imr[int(0.8 * len(self.imr)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]
       

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        # idx~[0~len(data)]
        data, tf, cfr_mce,imr, label = self.datas[idx], self.tfs[idx], self.cfr_mce[idx],self.imr[idx], self.labels[idx]
        data = torch.tensor(data)
        tf = torch.tensor(tf)
        cfr_mce = torch.tensor(cfr_mce)
        imr = torch.tensor(imr)
        label = torch.tensor(label)
        return data, tf, cfr_mce,imr, label