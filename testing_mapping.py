import numpy as np
import fonction_analytique as anat
import matplotlib.pyplot as plt
from os import path
import Mapping as mapp


image_path = path.join("images", "Irregulier.png")

T = mapp.reduction(image_path)


T1 = np.array(list(map(lambda x: 65 if x == 2 else x, T.flatten()))).reshape(T.shape)
T1 = np.array(list(map(lambda x: 80 if x == 8 else x, T1.flatten()))).reshape(T.shape)
T1 = np.array(list(map(lambda x: 55 if x == 0 else x, T1.flatten()))).reshape(T.shape)
T1 = np.array(list(map(lambda x: 60 if x == 1 else x, T1.flatten()))).reshape(T.shape)
T1 = np.array(list(map(lambda x: 70 if x == 7 else x, T1.flatten()))).reshape(T.shape)

anat.VisualizeMatrix(T1)