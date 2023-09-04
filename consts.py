
from abc import abstractmethod

class SVHN_HyperParameters:
    INPUT_SIZE = 3072  # 32x32x3
    HIDDEN_SIZE = 500
    NUM_CLASSES = 10
    NUM_EPOCHS = 20
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0001

class CIFAR10_HyperParameters:
    pass
