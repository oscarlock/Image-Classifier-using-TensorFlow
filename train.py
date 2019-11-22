import deepneuralnet as net
import numpy as np
from tflearn.data_utils import image_preloader
model = net.model
X, Y = image_preloader(target_path='./train', image_shape=(200, 200), mode='folder', grayscale=False, categorical_labels=True, normalize=True)
X = np.reshape(X, (-1, 200, 200, 3))

W, Z = image_preloader(target_path='./validate', image_shape=(200, 200), mode='folder', grayscale=False, categorical_labels=True, normalize=True)
W = np.reshape(W, (-1, 200, 200, 3))
model.fit(X, Y, n_epoch=250, validation_set=(W,Z), show_metric=True)
model.save('./ZeetrainedNet/final-model.tfl')
