import numpy as np
import Mapping as map

def dico_flag(matrice):
    rows, cols, _ = matrice.shape
    flag = {}
    for i in range(rows):
        for j in range(cols):
            if (matrice[i,j]==[255,255,255]).all() or (matrice[i,j]==[0,0,0]).all():
                flag[(i,j)]="exterieur"
            elif (matrice[i,j]==[255,0,0]).all():
                flag[(i,j)]="EDP"
            elif (matrice[i,j]==[0,255,0]).all():
                flag[(i,j)]="fantome"
            elif (matrice[i,j]==[0,0,255]).all():
                flag[(i,j)]="BC"
    
    return flag


matrix = np.array([[[255, 255, 255], [0, 255, 0], [0, 255, 0],[0, 255, 0], [255, 255, 255]],
                   [[0, 255, 0], [0, 0, 255], [0, 0, 255], [0, 0, 255], [0, 255, 0]],
                   [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 0, 255], [0, 255, 0]],
                   [[0, 255, 0], [0, 0, 255], [0, 0, 255], [0, 0, 255], [0, 255, 0]],
                   [[255, 255, 255], [0, 255, 0], [0, 255, 0], [0, 255, 0], [0,0,0]]])

matrix = map.mapping(map.png_to_rgb_matrix("exemple7.png"))

map.rgb_matrix_to_png(matrix)
print(dico_flag(matrix))