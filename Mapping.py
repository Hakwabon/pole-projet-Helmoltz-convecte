from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def png_to_rgb_matrix(file_path):
    # Ouvre l'image avec PIL
    image = Image.open(file_path)

    # Convertit l'image en mode RGB si elle ne l'est pas déjà
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Convertit l'image en matrice numpy
    rgb_matrix = np.array(image)
    # a,b,_=rgb_matrix.shape
    # m=0
    # for i in range(0,a):
    #     for j in range(0,b):
    #         if (rgb_matrix[i,j]!=[0,0,0]).all() and (rgb_matrix[i,j]!=[255,255,255]).all():
    #             rgb_matrix[i,j]=[255,255,255]
    #             m+=1     
    # print(m)
    return rgb_matrix


def rgb_matrix_to_png(rgb_matrix):
    """
    Cette fonction prend en entrée une matrice RGB et affiche l'image PNG
    correspondante en utilisant la bibliothèque Matplotlib.
    """
    # Convertit la matrice RGB en tableau Numpy
    np_array = np.array(rgb_matrix, dtype=np.uint8)

    # Crée une figure Matplotlib
    fig, ax = plt.subplots()

    # Affiche l'image à l'aide de la méthode "imshow"
    ax.imshow(np_array)

    # Affiche la figure
    # plt.show()


def bw_matrix_to_binary(matrix):
    """
    Cette fonction prend en entrée une matrice RGB en noir et blanc et renvoie
    la même matrice qui associe au noir la valeur 1 et au blanc la valeur 0.
    """
    binary_matrix = []
    for row in matrix:
        binary_row = []
        for pixel in row:
            # Si le pixel est noir, on ajoute 1 à la ligne binaire
            if pixel == (0, 0, 0):
                binary_row.append(1)
            # Sinon, on ajoute 0
            else:
                binary_row.append(0)
        binary_matrix.append(binary_row)

    return binary_matrix


def binary_matrix_to_bw(matrix):
    """
    Cette fonction prend en entrée une matrice binaire de 0 et de 1 et renvoie
    la matrice RGB associée où les pixels blancs ont une valeur RGB de
    (255, 255, 255) et les pixels noirs ont une valeur RGB de (0, 0, 0).
    """
    bw_matrix = []
    for row in matrix:
        bw_row = []
        for pixel in row:
            # Si le pixel vaut 1, on ajoute la couleur noire à la ligne binaire
            if pixel == 1:
                bw_row.append((0, 0, 0))
            # Sinon, on ajoute la couleur blanche
            else:
                bw_row.append((255, 255, 255))
        bw_matrix.append(bw_row)

    # Convertit la matrice RGB en tableau Numpy
    np_array = np.array(bw_matrix, dtype=np.uint8)

    return np_array





# La fonction prend en entrer une matrice rgb et detecte si c'est un point interieur, si c'est le cas il colorie cette case en rouge


def mapping_origine(matrix,):  # on considère que l'image de départ est juste noire et blanche et qu'on cherche à remplir le blanc avec du rouge
    rows, cols, _ = matrix.shape
    new_matrix = np.zeros_like(matrix)
    new_matrix[:, :, :] = matrix[:, :, :]
    d = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    file = [
        (rows // 2, 9*cols // 10)
    ]  # Le point seed du remplissage est le point centrale de l'image
    while file != []:
        i, j = file.pop(0)
        if (matrix[i, j] == [255, 255, 255]).all() and (
            new_matrix[i, j] == [255, 255, 255]
        ).all():
            new_matrix[i, j] = [255, 0, 0]
            for di, dj in d:
                ni = i + di
                nj = j + dj
                if (0 <= ni < rows and 0 <= nj < cols and (new_matrix[ni, nj] == [255, 255, 255]).all()):
                    file.append((ni, nj))
                if (0 <= ni < rows and 0 <= nj < cols and (matrix[ni, nj] == [0, 0, 0]).all()):
                    new_matrix[i, j] = [0, 0, 255]
                    # new_matrix[ni, nj] = [0, 255, 0]
                    # new_matrix[ni, nj] = [0, 0, 255]
    return new_matrix

def mapping(matrix,):  # on considère que l'image de départ est juste noire et blanche et qu'on cherche à remplir le blanc avec du rouge
    rows, cols, _ = matrix.shape
    new_matrix = np.zeros_like(matrix)
    new_matrix[:, :, :] = matrix[:, :, :]
    d = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    file = [
        (rows // 2, 9*cols // 10)
    ]  # Le point seed du remplissage est le point centrale de l'image
    # Initialisation du compteur
    total_pixels = rows * cols
    filled_pixels = 0
    while file != []:
        i, j = file.pop(0)
        if (matrix[i, j] == [255, 255, 255]).all() and (new_matrix[i, j] == [255, 255, 255]).all():
            new_matrix[i, j] = [255, 0, 0]
            filled_pixels += 1
            for di, dj in d:
                ni = i + di
                nj = j + dj
                if (0 <= ni < rows and 0 <= nj < cols and (new_matrix[ni, nj] == [255, 255, 255]).all()):
                    file.append((ni, nj))
                if (0 <= ni < rows and 0 <= nj < cols and (matrix[ni, nj] == [0, 0, 0]).all()):
                    new_matrix[i, j] = [0, 0, 255]
                    # new_matrix[ni, nj] = [0, 255, 0]
                    # print("else")
                    
                    
                    # # result =  int((new_matrix[ni+1, nj][-1]-1)/2+ (new_matrix[ni-1, nj][-1]-1)/2+(new_matrix[ni, nj+1][-1]-1)/2+ (new_matrix[ni, nj-1][-1]-1)/2+2)
                    # result =  int((matrix[i+1, j][-1]-1)/2+ (matrix[i-1, j][-1]-1)/2+(matrix[i, j+1][-1]-1)/2+ (matrix[i, j-1][-1]-1)/2+2)
                    # result2 = int((matrix[ni+1, nj][-1]-1)/2+ (matrix[ni-1, nj][-1]-1)/2+(matrix[ni, nj+1][-1]-1)/2+ (matrix[ni, nj-1][-1]-1)/2+2)
                    # if result == 255 :
                    #     print(i,j)
                    #     new_matrix[ni, nj] = [0, 0, 255]
        if filled_pixels % 1000 == 0:
            print(f"Progress: {filled_pixels} pixels filled out of {total_pixels}")

    print(f"Final filled pixels: {filled_pixels} out of {total_pixels}")
    
    
    j = 7*cols/10 // 2
    for n in range(rows):
        for k in range(cols):
            if (new_matrix[n, k] == [0,0,255]).all():
                if k > j:
                    j = k
                    
    i = rows//2
    n,k = i+1,j
    p,m = n,k

    
    

    while [i,j] != [n,k] :

        if (new_matrix[i,j-1]==[0,0,255]).all():
            j-=1
            new_matrix[i,j]=[0,255,0]
            direction = 1
            a = 1
        elif (new_matrix[i,j+1]==[0,0,255]).all():
            j+=1
            new_matrix[i,j]=[0,255,0]
            direction = 2
            a = 1
        elif (new_matrix[i-1,j]==[0,0,255]).all():
            i-=1
            new_matrix[i,j]=[0,255,0]
            direction = 4
            a = 1
        elif (new_matrix[i+1,j]==[0,0,255]).all():
            i+=1
            new_matrix[i,j]=[0,255,0]
            direction = 3
            a = 1
        

        elif direction == 3 and a == 1:
            i+=1
            new_matrix[i,j]=[0,255,0]
            a = 2
        elif direction == 4 and a == 1:
            i-=1
            new_matrix[i,j]=[0,255,0]
            a = 2
        elif direction == 2 and a == 1:
            j+=1
            new_matrix[i,j]=[0,255,0]
            a = 2
        elif direction == 1 and a == 1:
            j-=1
            new_matrix[i,j]=[0,255,0]
            a = 2
        
        if [i,j]==[p,m]:
            print("break")
            break
        p,m = i,j

    new_matrix[n-1,k]=[0,255,0]
    return new_matrix

# rgb_matrix_to_png(mapping(png_to_rgb_matrix("exemple7.png")))


# Mapping qui remplie la matrice des flags
# CDH CDB CGH CGB indiquent que le point frontiere est un 'coin droit haut', un 'coin droit bas', un 'coin gauche haut' ou un 'coin gauche bas'
def color_to_flag_opti(rgb_matrix):
    rows, cols, _ = rgb_matrix.shape
    flag_matrix = np.empty((rows, cols))
    for i in range(0, rows):
        for j in range(0, cols):
            if (rgb_matrix[i, j] == [255, 255, 255]).all():
                flag_matrix[i, j] = 0
            elif (rgb_matrix[i, j] == [0, 0, 0]).all():
                flag_matrix[i, j] = 0
            elif (rgb_matrix[i, j] == [0, 255, 0]).all():
                flag_matrix[i, j] = 1
            elif (rgb_matrix[i, j] == [0, 0, 255]).all():
                flag_matrix[i, j] = 1
            elif (rgb_matrix[i, j] == [255, 0, 0]).all():
                flag_matrix[i, j] = 2
            else:
                print("l'image n'est pas en noir [0,0,0] est blanc [255,255,255]")
                # print("i,j",i,j)
                print(rgb_matrix[i,j])
    return flag_matrix

def mapping_BC_opti(Matrice):
    n, p = Matrice.shape
    j_min = p // 2
    j_max = p // 2
    for i in range(n):
        for j in range(p):
            if Matrice[i, j] == 1:
                if j > j_max:
                    j_max = j
                elif j < j_min:
                    j_min = j

    for i in range(n):
        if Matrice[i, j_min]==1:
            #BC out, sorti du reacteur
            Matrice[i, j_min] = 7
        if Matrice[i, j_max]==1:
            #BC in, entré du reacteur
            Matrice[i, j_max] = 8

    return Matrice

def mapping_BC_opti_2(Matrice):
    print("mapping BC")
    n, p = Matrice.shape
    j_min = p // 2
    j_max = 7*p/10 // 2
    for i in range(n):
        for j in range(p):
            if Matrice[i, j] == 1:
                if j > j_max:
                    j_max = j
                elif j < j_min:
                    j_min = j

    i_min_l = n//2
    i_max_l = n//2
    for i in range(n):
        if Matrice[i, j_min]==1:
            #BC out, sorti du reacteur
            Matrice[i, j_min] = 7
            if i<i_min_l:
                i_min_l = i
            elif i>i_max_l:
                i_max_l = i
        if Matrice[i, j_max]==1:
            #BC in, entré du reacteur
            Matrice[i, j_max] = 8
        print(round(i/n*100),"%", 1)

    i_min_r = i_min_l
    i_max_r = i_max_l
    Matrice[i_min_l,j_min]=1
    Matrice[i_max_l,j_min]=1

    upper_borders = [[i_min_r,j_min]]  # Liste pour stocker les indices des bords haut
    lower_borders = [[i_max_r,j_min]]  # Liste pour stocker les indices des bords bas
    j = j_min

    while j<j_max-1:
        if Matrice[i_min_r,j+1]==(1 or 8):
            j+=1
            upper_borders.append([i_min_r,j])
        elif Matrice[i_min_r+1,j]==(1 or 8):
            i_min_r+=1
            upper_borders.append([i_min_r,j])
        elif Matrice[i_min_r-1,j]==(1 or 8):
            i_min_r-=1
            upper_borders.append([i_min_r,j])
        Matrice[i_min_r,j] = 5
        print(round(j/(j_max-1)*100),"%",2)

    j = j_min

    while j<j_max-1:
        if Matrice[i_max_r,j+1]== (1 or 8):
            j+=1
            lower_borders.append([i_max_r,j])
        elif Matrice[i_max_r+1,j]==(1 or 8):
            i_max_r+=1
            lower_borders.append([i_max_r,j])
        elif Matrice[i_max_r-1,j]==(1 or 8):
            i_max_r-=1
            lower_borders.append([i_max_r,j])

        Matrice[i_max_r,j] = 5
        print(round(j/(j_max-1)*100),"%",3)

    j = j_min+1
    i = i_min_r+1
    i_min = i_max_l

    Matrice[i_min_l,j_min] = 71
    Matrice[i_min_r,j_max-1] =73
    Matrice[i_max_l,j_min] = 76
    Matrice[i_max_r,j_max-1] =78

    # print(Matrice)
    while Matrice[i_min-1,j]!=1 and Matrice[i_min-1,j]!=5:
        i_min-=1

    while Matrice[i+1,j]!=1 and Matrice[i+1,j]!=5 :
        i+=1

    inside_borders = [[i,j]]
    
    if len(inside_borders)>1:
        Matrice[i,j]=77
        while i!=i_min or j != j_min+1:
            if Matrice[i,j+1]== 1:
                j+=1
                inside_borders.append([i,j])
            elif Matrice[i+1,j]==1:
                i+=1
                inside_borders.append([i,j])
            elif Matrice[i-1,j]==1:
                i-=1
                inside_borders.append([i,j])
            elif Matrice[i,j-1]==1:
                j-=1
                inside_borders.append([i,j])
            Matrice[i,j] = 5

        Matrice[i,j]=72
    # Matrice[i_min_l,j_min] = 72
    # Matrice[i_min_r,j_max-1] =72
    # Matrice[i_max_l,j_min] = 77
    # Matrice[i_max_r,j_max-1] =72
    wall_direction = 7

    for i in range(len(upper_borders) - 2):
        # Coordonnées des points consécutifs
        point1 = upper_borders[i]
        point2 = upper_borders[i + 1]
        point3 = upper_borders[i + 2]
        
        # Comparer les coordonnées
        if (point1[0] == point2[0] == point3[0]):
            Matrice[point2[0],point2[1]]=72
        elif (point1[1] == point2[1] == point3[1] ):
            Matrice[point2[0],point2[1]]=wall_direction
        elif (point1[0] == point2[0] != point3[0] ):
            if point1[0]<point3[0]:
                Matrice[point2[0],point2[1]]=73
                wall_direction=75
            elif point1[0]>point3[0]:
                Matrice[point2[0],point2[1]]=71
                wall_direction=74
        elif (point1[1] == point2[1] != point3[1] ):
            if point1[0]<point3[0]:
                Matrice[point2[0],point2[1]]=73
            elif point1[0]>point3[0]:
                Matrice[point2[0],point2[1]]=71

    for i in range(len(lower_borders) - 2):
        # Coordonnées des points consécutifs
        point1 = lower_borders[i]
        point2 = lower_borders[i + 1]
        point3 = lower_borders[i + 2]
        
        # Comparer les coordonnées
        if (point1[0] == point2[0] == point3[0]):
            Matrice[point2[0],point2[1]]=77
        elif (point1[1] == point2[1] == point3[1] ):
            Matrice[point2[0],point2[1]]=wall_direction
        elif (point1[0] == point2[0] != point3[0] ):
            if point1[0]<point3[0]:
                Matrice[point2[0],point2[1]]=76
                wall_direction=74
            elif point1[0]>point3[0]:
                Matrice[point2[0],point2[1]]=78
                wall_direction=74
        elif (point1[1] == point2[1] != point3[1] ):
            if point1[0]<point3[0]:
                Matrice[point2[0],point2[1]]=76
            elif point1[0]>point3[0]:
                Matrice[point2[0],point2[1]]=78

    wall_direction = 77
    for i in range(len(inside_borders) - 2):
        # Coordonnées des points consécutifs
        point1 = inside_borders[i]
        point2 = inside_borders[i + 1]
        point3 = inside_borders[i + 2]
        
        # Comparer les coordonnées
        if (point1[0] == point2[0] == point3[0]):
            Matrice[point2[0],point2[1]]=wall_direction
        elif (point1[1] == point2[1] == point3[1] ):
            Matrice[point2[0],point2[1]]=wall_direction
        elif (point1[0] == point2[0] != point3[0] ):
            if point1[0]<point3[0]:
                if point1[1]<point3[1]:
                    Matrice[point2[0],point2[1]]=76
                elif point1[1]>point3[1]:
                    Matrice[point2[0],point2[1]]=71
                wall_direction=74
            elif point1[0]>point3[0]:
                if point1[1]>point3[1]:
                    Matrice[point2[0],point2[1]]=73
                elif point1[1]<point3[1]:
                    Matrice[point2[0],point2[1]]=78
                wall_direction=75
        elif (point1[1] == point2[1] != point3[1] ):
            if point1[0]<point3[0]:
                if point1[1]<point3[1]:
                    Matrice[point2[0],point2[1]]=76
                elif point1[1]>point3[1]:
                    Matrice[point2[0],point2[1]]=71
                wall_direction = 72
            elif point1[0]>point3[0]:
                if point1[1]>point3[1]:
                    Matrice[point2[0],point2[1]]=73
                elif point1[1]<point3[1]:
                    Matrice[point2[0],point2[1]]=78
                wall_direction = 72

    return Matrice

# print(mapping_BC(color_to_flag(mapping(png_to_rgb_matrix("exemple7.png")))))

def reduction(image):
    matrix = mapping_BC_opti_2(color_to_flag_opti(mapping(png_to_rgb_matrix(image))))
    print("réduction")
    to_remove = {"top":0,"bottom":0,"right":0,"left":0}
    N,J=matrix.shape
    i = 0
    a = 1
    while i < N and a ==1:
        ligne_to_remove = True
        for j in range(J):
            if matrix[i,j]!=0:
                ligne_to_remove = False
        if ligne_to_remove == False:
            a = N
        i+=1
    to_remove["top"]=i-1

    i = N-1
    a = 1
    while i > 0 and a ==1:
        ligne_to_remove = True
        for j in range(J):
            if matrix[i,j]!=0:
                ligne_to_remove = False
        if ligne_to_remove == False:
            a = 0
        i-=1
    to_remove["bottom"]=i+2

    j = 0
    a = 1
    while j < J and a == 1:
        ligne_to_remove = True
        for i in range(N):
            if matrix[i,j]!=0:
                ligne_to_remove = False
        if ligne_to_remove == False:
            a = J
        j+=1
    to_remove["left"]=j-1

    j = J-1
    a = 1
    while j > 0 and a == 1 :
        ligne_to_remove = True
        for i in range(N):
            if matrix[i,j]!=0:
                ligne_to_remove = False
        if ligne_to_remove == False:
            a = 0
        j-=1
    to_remove["right"]=j+2

    matrix = matrix[to_remove["top"]:to_remove["bottom"],to_remove["left"]:to_remove["right"]]

    return matrix


# print(dico)

# matrice_reacteur = mapping_BC(color_to_flag(mapping(png_to_rgb_matrix("exemple7.png"))))
#    mapping(png_to_rgb_matrix("exemple6.png"))))
# rgb_matrix_to_png(mapping(png_to_rgb_matrix("exemple6.png")))
# rgb_matrix_to_png(mapping(png_to_rgb_matrix("exemple5.png")))

# print(matrice_reacteur)

# n, p = np.shape(matrice_reacteur)
# Fonction de bijection !!
# print(mapping_BC_opti(color_to_flag_opti(mapping(png_to_rgb_matrix("exemple7.png")))))
# print(reduction("images/Irregulier.png"))

image = "images/réacteur2_mini.png"
# image = "images/réacteur2_mini.png"
# print(rgb_matrix_to_png(mapping_origine(png_to_rgb_matrix(image))))
# print(rgb_matrix_to_png(mapping(png_to_rgb_matrix(image))))
# print(mapping_BC_opti_2(color_to_flag_opti(mapping(png_to_rgb_matrix(image)))))
# print(mapping(png_to_rgb_matrix("images/Irregulier.png")))
# print(rgb_matrix_to_png(reduction(image)))
plt.show()

# def vectorise(i, j, p):
#     return p * i + j


# def matricise(k, p):
#     return (k // p, k % p)
