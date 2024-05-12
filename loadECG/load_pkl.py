import pickle
import numpy as np


def readData(dataSet):
    signal, t_f, cfr_mce,imr, label = [], [], [], [],[]
    for index in dataSet:
        signal.append(index['heartbeat'])
        t_f.append(index['T_f'])
        cfr_mce.append(index['cfr_mce'])
        imr.append(index['imr'])
        label.append(index['label'])
    signal = np.asarray(signal).astype(np.float32)
    t_f = np.asarray(t_f).astype(np.float32)
    cfr_mce = np.asarray(cfr_mce).astype(np.float32)
    imr = np.asarray(imr).astype(np.float32)
    label = np.asarray(label)
    return signal, t_f, cfr_mce,imr, label


def loadData(filename):
    with open(filename, "rb") as f:
        train_heartbeats, test_heartbeats = pickle.load(f)

    trainSet = train_heartbeats
    testSet = test_heartbeats

    signal_train, tf_train, label_train = readData(trainSet)
    signal_test, tf_test, label_test = readData(testSet)
    print('Data loading...')

    return signal_train, tf_train, label_train, signal_test, tf_test, label_test


def loadData_noSplit(filename):
    with open(filename, "rb") as f:
        heartbeats = pickle.load(f)
    DbSet = heartbeats
    signal, t_f, cfr_mce,imr, label = readData(DbSet)
    return signal, t_f, cfr_mce,imr, label


def readPtb(dataSet):
    signal, t_f, label, group = [], [], [], []
    for index in dataSet:
        signal.append(index['signal'])
        t_f.append(index['C_TWA'])
        label.append(index['label'])
        group.append(index['record'])
    signal = np.asarray(signal).astype(np.float32)
    t_f = np.asarray(t_f).astype(np.float32)
    # label = np.asarray(label).astype(np.float32)
    # label = label.astype(np.int32)
    group = np.asarray(group)
    return signal, t_f, label, group


def load_ptb(filename):
    with open(filename, 'rb') as f:
        train_data, test_data = pickle.load(f)
    DbSet = train_data + test_data
    signal, t_f, label, group = readPtb(DbSet)
    return signal, t_f, label, group
