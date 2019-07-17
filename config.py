NAN = -1.0
LINE_LENGTH = 200


class DataConfig:
    DIRECTORY = '../Database/'

    GLASS = '../Database/Glass/data.txt'
    HAMBERMAN = '../Database/Hamberman/data.txt'
    IRIS = '../Database/Iris/data.txt'
    MUSK = '../Database/Musk/data.txt'
    WINE = '../Database/Wine/data.txt'
    YEAST = '../Database/Yeast/data.txt'

    NAME = ['Glass',
            'Hamberman'
            'Iris',
            'Musk',
            'Wine',
            'Yeast']

    MISSING_RATIO = 0.05


class FCMParam:
    ERROR = 1e-5
    MAX_ITR = 1000


class SVRParam:
    C = 1
    EP = 1e-5


class GAParam:
    GENERATIONS = 40
    POPULATION_SIZE = 20
    CF = 0.6
    MF = 0.03
