import math
import random
import warnings

import numpy as np
import pandas
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import MinMaxScaler

from config import DataConfig, NAN, LINE_LENGTH

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
np.set_printoptions(linewidth=LINE_LENGTH)


# Normalize the given data with min-max normalization
def normalize_data(data):
    transformer = MinMaxScaler()
    transformer.fit(data)
    data = transformer.transform(data)
    return data


# Add missing value to  sample dataset with specific ratio
def add_missing_value(data, ratio):
    n = math.ceil(ratio * len(data))
    samples = random.sample(range(len(data)), n)
    for index in samples:
        i = random.randint(0, len(data[index]) - 1)
        data[index][i] = NAN

    return data


class Driver:
    def __init__(self):
        self.__dataset = {
            'Glass': Driver.__load_glass_data(),
            'Hamberman': Driver.__load_hamberman_data(),
            'Iris': Driver.__load_iris_data(),
            'Musk': Driver.__load_musk_data(),
            'Wine': Driver.__load_wine_data(),
            'Yeast': Driver.__load_yeast_data()
        }

        print('Datasets Loaded Successfully!')

    @staticmethod
    def __load_glass_data():
        data = np.array(pandas.read_csv(filepath_or_buffer=DataConfig.GLASS, sep=',', header=None))
        labels = [int(value) - 1 for value in data[:, len(data[0]) - 1]]
        data = normalize_data(data)
        return {'data': np.array(data[:, 1:len(data[0]) - 1]), 'labels': np.array(labels)}

    @staticmethod
    def __load_hamberman_data():
        data = np.array(pandas.read_csv(filepath_or_buffer=DataConfig.HAMBERMAN, sep=',', header=None))
        labels = [int(value) - 1 for value in data[:, len(data[0]) - 1]]
        data = normalize_data(data)
        return {'data': np.array(data[:, :len(data[0]) - 1]), 'labels': np.array(labels)}

    @staticmethod
    def __load_iris_data():
        data = np.array(pandas.read_csv(filepath_or_buffer=DataConfig.IRIS, sep=',', header=None))
        target = dict([(y, x + 1) for x, y in enumerate(sorted(set(data[:, len(data[0]) - 1])))])
        labels = [int(target[x]) - 1 for x in data[:, len(data[0]) - 1]]
        data = data[:, :len(data[0]) - 1]
        data = normalize_data(data)
        return {'data': np.array(data), 'labels': np.array(labels)}

    @staticmethod
    def __load_musk_data():
        data = np.array(pandas.read_csv(filepath_or_buffer=DataConfig.MUSK, sep=',', header=None))
        labels = [int(value) for value in data[:, len(data[0]) - 1]]
        data = data[:, 2:len(data[0]) - 1]
        data = normalize_data(data)
        return {'data': np.array(data), 'labels': np.array(labels)}

    @staticmethod
    def __load_wine_data():
        data = np.array(pandas.read_csv(filepath_or_buffer=DataConfig.WINE, sep=',', header=None))
        labels = [int(value) - 1 for value in data[:, 0]]
        data = normalize_data(data)
        return {'data': np.array(data[:, 1:]), 'labels': np.array(labels)}

    @staticmethod
    def __load_yeast_data():
        data = np.array(pandas.read_csv(filepath_or_buffer=DataConfig.YEAST, sep=r'\s+', header=None))
        target = dict([(y, x + 1) for x, y in enumerate(sorted(set(data[:, len(data[0]) - 1])))])
        labels = [int(target[x]) - 1 for x in data[:, len(data[0]) - 1]]
        data = data[:, 1:len(data[0]) - 1]
        data = normalize_data(data)
        return {'data': np.array(data), 'labels': np.array(labels)}

    def get_dataset(self):
        return self.__dataset
