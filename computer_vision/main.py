import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers

from utils import config
from utils import modelUtils
from utils import trainUtils

# Setting seed for reproducibiltiy
keras.utils.set_random_seed(config.SEED)


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")





# Run experiments with the vanilla ViT
vit = modelUtils.create_vit_classifier(x_train, vanilla=True)
history = trainUtils.run_experiment(vit, x_train, y_train, x_test, y_test)

# Run experiments with the Shifted Patch Tokenization and
# Locality Self Attention modified ViT
vit_sl = modelUtils.create_vit_classifier(x_train, vanilla=False)
history = trainUtils.run_experiment(vit_sl, x_train, y_train, x_test, y_test)