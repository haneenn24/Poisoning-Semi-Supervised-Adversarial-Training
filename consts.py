
from abc import abstractmethod

class SVHN_HyperParameters:
    INPUT_SIZE = 3072  # 32x32x3
    HIDDEN_SIZE = 500
    NUM_CLASSES = 10
    NUM_EPOCHS = 2
    BATCH_SIZE = 100
    LEARNING_RATE = 0.001

class CIFAR10_HyperParameters:
    pass
