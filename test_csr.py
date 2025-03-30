import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv
# import matrice_operateurs as operateur
from parameters import *
import matplotlib.pyplot as plt
import Mapping as mapp
import fonction_analytique as anat
import affichage as aff
import tkinter as tk

# Matrice = np.array([[1,0,0,1,1,1,0,0],[1,1,1,1,0,1,1,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[1,1,1,1,0,1,1,0],[1,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0]])
Matrice = np.array([[0,0,0,0,0,0,0,1,1,1,1,0],
                    [1,1,1,1,0,0,0,1,0,0,1,0],
                    [1,0,0,1,1,1,1,1,0,0,1,0],
                    [1,0,0,0,0,0,0,0,0,0,1,0],
                    [1,0,0,0,0,0,0,0,0,0,1,0],
                    [1,0,0,0,0,0,0,0,0,0,1,0],
                    [1,0,0,0,0,0,0,0,0,0,1,0],
                    [1,1,1,1,1,1,1,1,0,0,1,0],
                    [0,0,0,0,0,0,0,1,1,1,1,0]])

Matrix = np.array([[0,0,0,0,0,0,0,1,1,1,1,0],
                    [1,1,1,1,0,0,0,1,0,0,1,0],
                    [1,0,0,1,1,1,1,1,0,0,1,0],
                    [1,0,0,0,0,0,0,0,0,0,1,0],
                    [1,1,1,1,1,1,1,1,1,0,1,0],
                    [1,0,0,0,0,0,0,0,1,0,1,0],
                    [1,1,1,0,1,1,1,0,1,0,1,0],
                    [1,0,1,0,1,0,1,0,1,0,1,0],
                    [1,0,1,1,1,0,1,1,1,0,1,0],
                    [1,0,0,0,0,0,0,0,0,0,1,0],
                    [1,1,1,1,1,1,1,0,0,0,1,0],
                    [0,0,0,0,0,0,1,1,1,1,1,0]])




Matrix = np.zeros((100, 100))
Lx = 100
Ly =50
# display = True

multiplier_4_increasing_resolution = 2


# nodes_y, nodes_x = T.shape

# metric = anat.Metric(Lx, Ly, nodes_x, nodes_y)

# T = anat.setting_matrix_domain(origin = "rect", nodes_x = Lx * multiplier_4_increasing_resolution + 1, nodes_y = Ly * multiplier_4_increasing_resolution + 1)

# nodes_y, nodes_x = T.shape

# metric = anat.Metric(Lx, Ly, nodes_x, nodes_y)

# anat.plot_analytical_solution(T,f = None,metric=metric,a=1)
# aff.traitement()
image = "images/réacteur2_mini.png"
# image = "images/Irregulier.png"
mat = mapp.reduction(image)
k,n = mat.shape
for i in range(k):
    for j in range(n):
        if mat[i,j] == 2 or mat[i,j] == 1:
            mat[i,j] = 70
        elif mat[i,j] == 8:
            mat[i,j] = 68
        elif mat[i,j] == 0:
            mat[i,j] = 66
        elif mat[i,j] == 7:
            mat[i,j] = 68
anat.VisualizeMatrix(mat)
# anat.VisualizeMatrix(mapp.color_to_flag_opti(mapp.mapping(mapp.png_to_rgb_matrix(image))))

plt.show()



