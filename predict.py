import deepneuralnet as net
import numpy as np
from tflearn.data_utils import image_preloader
model = net.model
path_to_model = './ZeetrainedNet/final-model.tfl'
model.load(path_to_model)
X, Y = image_preloader(target_path='./validate', image_shape=(200,200), mode='folder', grayscale=False, categorical_labels=True, normalize=True)
X = np.reshape(X, (-1, 200, 200, 3))
for i in range(0, len(X)):
 iimage = X[i]
 icateg = Y[i]
 result = model.predict([iimage])[0]
 prediction = result.tolist().index(max(result))
 reality = icateg.tolist().index(max(icateg))
 if prediction == reality:
    print("image %d CORRECT " % i, end='')
 else:
    print("image %d WRONG " % i, end='')
 print(result)
