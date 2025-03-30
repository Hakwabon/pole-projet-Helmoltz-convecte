import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Mapping as mapp
from parameters import *
import fonction_analytique as anat
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv

def laplacien(T):

    nx,ny=T.shape

    # Création de la matrice du Laplacien
    laplacian_matrix = np.zeros((nx*ny, nx*ny),dtype=complex)
    

    # Remplissage de la matrice du Laplacien
    for i in range(nx):
        for j in range(ny):
            k = i + j * nx  # Indice de la cellule (i, j)

            if T[i,j]==2:

                laplacian_matrix[k, k] = -2.0 / (hx*hx) - 2.0 / (hy*hy)
                
                if i > 0:
                    laplacian_matrix[k, k - 1] = 1.0 / (hx*hx)  # Voisin à gauche
                if i < nx - 1:
                    laplacian_matrix[k, k + 1] = 1.0 / (hx*hx)  # Voisin à droite
                if j > 0:
                    laplacian_matrix[k, k - nx] = 1.0 / (hy*hy)  # Voisin en haut
                if j < ny - 1:
                    laplacian_matrix[k, k + nx] = 1.0 / (hy*hy)  # Voisin en bas
                    
    # Affichage de la matrice du Laplacien
    # print("Matrice du Laplacien :")
    # print(laplacian_matrix)
    # Flatten la matrice Force en un vecteur colonne
    return laplacian_matrix

def laplacien_csr(T):
    nx, ny = T.shape

    # Listes pour stocker les valeurs non nulles de la matrice
    data = []
    row_indices = []
    col_indices = []

    # Taille de la grille
    grid_size = nx * ny

    # Paramètres de discrétisation
    hx = 1  # Pas de discrétisation en x
    hy = 1  # Pas de discrétisation en y

    # Remplissage de la matrice du Laplacien
    for i in range(nx):
        for j in range(ny):
            k = i + j * nx  # Indice de la cellule (i, j)

            if T[i, j] == 2:
                # Élément diagonal
                data.append(-2.0 / (hx * hx) - 2.0 / (hy * hy))
                row_indices.append(k)
                col_indices.append(k)

                # Voisin à gauche
                if i > 0:
                    data.append(1.0 / (hx * hx))
                    row_indices.append(k)
                    col_indices.append(k - 1)

                # Voisin à droite
                if i < nx - 1:
                    data.append(1.0 / (hx * hx))
                    row_indices.append(k)
                    col_indices.append(k + 1)

                # Voisin en haut
                if j > 0:
                    data.append(1.0 / (hy * hy))
                    row_indices.append(k)
                    col_indices.append(k - nx)

                # Voisin en bas
                if j < ny - 1:
                    data.append(1.0 / (hy * hy))
                    row_indices.append(k)
                    col_indices.append(k + nx)

    # Conversion des listes en tableaux numpy
    data = np.array(data,dtype=complex)
    row_indices = np.array(row_indices)
    col_indices = np.array(col_indices)

    # Création de la matrice CSR
    laplacian_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(grid_size, grid_size))

    return laplacian_matrix

def Force(T):
    nx,ny=T.shape
    b = np.zeros((nx*ny,1),dtype=complex)
    for i in range(nx):
        for j in range(ny):
            k = i + j * nx
            b[k,0]=1
    return b

def identite(T):
    nx,ny=T.shape

    # Création de la matrice du Laplacien
    matrix = np.zeros((nx*ny, nx*ny),dtype=complex)
    
    # Remplissage de la matrice du Laplacien
    for i in range(nx):
        for j in range(ny):
            k = i + j * nx  # Indice de la cellule (i, j)
            if T[i,j]==2:
                matrix[k, k] = k0*k0

    return matrix

def identite_csr(T):
    nx, ny = T.shape

    # Listes pour stocker les valeurs non nulles de la matrice
    data = []
    row_indices = []
    col_indices = []

    # Remplissage de la matrice identité
    for i in range(nx):
        for j in range(ny):
            k = i + j * nx  # Indice de la cellule (i, j)
            if T[i, j] == 2:
                # Ajout de la valeur non nulle sur la diagonale
                data.append(k0 * k0)
                row_indices.append(k)
                col_indices.append(k)

    # Conversion des listes en tableaux numpy
    data = np.array(data, dtype=complex)
    row_indices = np.array(row_indices)
    col_indices = np.array(col_indices)

    # Création de la matrice CSR
    identity_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(nx * ny, nx * ny))

    return identity_matrix

def BC_robin_0(T):
    nx,ny=T.shape

    # Création de la matrice du Laplacien
    matrix = np.zeros((nx*ny, nx*ny),dtype=complex)
    
    # Remplissage de la matrice du Laplacien
    for i in range(nx):
        for j in range(ny):
            k = i + j * nx  # Indice de la cellule (i, j)
            if T[i,j]==7:

                matrix[k, k] = -2.0 / (hx*hx) - 2.0 / (hy*hy) + k0*k0
                
                if i == 0:
                    matrix[k, k+1 ] = 1.0 / (hx*hx)  # Voisin à gauche
                if i == nx - 1:
                    matrix[k, k-1 ] = 1.0 / (hx*hx)  # Voisin à droite
                if j == 0:
                    matrix[k, k+nx ] = 1.0 / (hy*hy)  # Voisin en haut
                if j == ny - 1:
                    matrix[k, k-nx] = 1.0 / (hy*hy)  # Voisin en bas
            



    return matrix

def BC_robin(T):
    nx,ny=T.shape

    # Création de la matrice 
    matrix = np.zeros((nx*ny, nx*ny),dtype=complex)
    
    # Remplissage de la matrice du Laplacien
    for i in range(nx):
        for j in range(ny):
            k = i + j * nx  # Indice de la cellule (i, j)
            if T[i,j]==7:

                matrix[k, k] = (-2 / (hx*hx) - 2*complex(1,-k0*hy) / (hy*hy) + k0*k0)
                
                if i > 0:
                    matrix[k, k-1 ] = 1.0 / (hx*hx)  # Voisin à gauche
                if i < nx - 1:
                    matrix[k, k+1 ] = 1.0 / (hx*hx)  # Voisin à droite
                if j > 0:
                    matrix[k, k-nx ] = 2.0 / (hy*hy)  # Voisin en haut
                if j < ny - 1:
                    matrix[k, k+nx] = 2.0 / (hy*hy)  # Voisin en bas
            
    return matrix

def BC_dirichlet(T):

    nx,ny=T.shape

    matrix = np.zeros((nx*ny, nx*ny),dtype=complex)

    # Remplissage de la matrice
    for i in range(nx):
        for j in range(ny):
            k = i + j * nx  # Indice de la cellule (i, j)
            if T[i,j]==1:
                matrix[k, k] = 1
                # matrix[k, k] = -2.0 / (hx*hx) - 2.0 / (hy*hy) + k0*k0
                
                # if i == 0:
                #     matrix[k, k+1 ] = 0  # Voisin à gauche
                # if i == nx - 1:
                #     matrix[k, k-1 ] = 0  # Voisin à droite
                # if j == 0:
                #     matrix[k, k+nx ] = 0  # Voisin en haut
                # if j == ny - 1:
                #     matrix[k, k-nx] = 0  # Voisin en bas

    return matrix

def BC_onde(T):
    nx,ny=T.shape
    b = np.zeros((nx*ny,1),dtype=complex)
    matrix = np.zeros((nx*ny, nx*ny),dtype=complex)
    for i in range(nx):
        for j in range(ny):
            k = i + j * nx

            if T[i,j]==8:
                b[k,0]=-np.exp(complex(0,k0*(ny-j)))*i*(nx-i)/(hy*hy)

                matrix[k, k] = -2.0 / (hx*hx) - 2.0 / (hy*hy) + k0*k0
                
                if i > 0:
                    matrix[k, k - 1] = 1.0 / (hx*hx)  # Voisin à gauche
                if i < nx - 1:
                    matrix[k, k + 1] = 1.0 / (hx*hx)  # Voisin à droite
                if j > 0:
                    matrix[k, k - nx] = 1.0 / (hy*hy)  # Voisin en haut
                if j < ny - 1:
                    matrix[k, k + nx] = 1.0 / (hy*hy)  # Voisin en bas
    return matrix,b

def force_test(T):
    nx,ny=T.shape
    b = np.zeros((nx*ny,1),dtype=complex)
    
    for i in range(nx):
        for j in range(ny):
            k = i + j * nx
            if T[i,j]==2:
                b[k,0]=-2*np.exp(complex(0,k0*(ny-j)))
    
    b_csr = csr_matrix(b)            
    
    return b_csr

def laplacien_csr(T):
    nx, ny = T.shape

    # Listes pour stocker les valeurs non nulles de la matrice
    data = []
    row_indices = []
    col_indices = []

    # Taille de la grille
    grid_size = nx * ny

    # Paramètres de discrétisation
    hx = 1  # Pas de discrétisation en x
    hy = 1  # Pas de discrétisation en y

    # Remplissage de la matrice du Laplacien
    for i in range(nx):
        for j in range(ny):
            k = i + j * nx  # Indice de la cellule (i, j)

            if T[i, j] == 2:
                # Élément diagonal
                data.append(-2.0 / (hx * hx) - 2.0 / (hy * hy))
                row_indices.append(k)
                col_indices.append(k)

                # Voisin à gauche
                if i > 0:
                    data.append(1.0 / (hx * hx))
                    row_indices.append(k)
                    col_indices.append(k - 1)

                # Voisin à droite
                if i < nx - 1:
                    data.append(1.0 / (hx * hx))
                    row_indices.append(k)
                    col_indices.append(k + 1)

                # Voisin en haut
                if j > 0:
                    data.append(1.0 / (hy * hy))
                    row_indices.append(k)
                    col_indices.append(k - nx)

                # Voisin en bas
                if j < ny - 1:
                    data.append(1.0 / (hy * hy))
                    row_indices.append(k)
                    col_indices.append(k + nx)

    # Conversion des listes en tableaux numpy
    data = np.array(data,dtype=complex)
    row_indices = np.array(row_indices)
    col_indices = np.array(col_indices)

    # Création de la matrice CSR
    laplacian_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(grid_size, grid_size))

    return laplacian_matrix

def identite_csr(T):
    nx, ny = T.shape

    # Listes pour stocker les valeurs non nulles de la matrice
    data = []
    row_indices = []
    col_indices = []

    # Remplissage de la matrice identité
    for i in range(nx):
        for j in range(ny):
            k = i + j * nx  # Indice de la cellule (i, j)
            if T[i, j] == 2:
                # Ajout de la valeur non nulle sur la diagonale
                data.append(k0 * k0)
                row_indices.append(k)
                col_indices.append(k)

    # Conversion des listes en tableaux numpy
    data = np.array(data, dtype=complex)
    row_indices = np.array(row_indices)
    col_indices = np.array(col_indices)

    # Création de la matrice CSR
    identity_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(nx * ny, nx * ny))

    return identity_matrix

def BC_robin_csr(T):
    nx, ny = T.shape

    # Listes pour stocker les valeurs non nulles de la matrice
    data = []
    row_indices = []
    col_indices = []

    # Remplissage de la matrice du Laplacien
    for i in range(nx):
        for j in range(ny):
            k = i + j * nx  # Indice de la cellule (i, j)
            if T[i, j] == 7:
                diagonal_value = -2 / (hx * hx) - 2 * complex(1, -k0 * hy) / (hy * hy) + k0 * k0
                data.append(diagonal_value)
                row_indices.append(k)
                col_indices.append(k)

                if i > 0:
                    data.append(1.0 / (hx * hx))  # Voisin à gauche
                    row_indices.append(k)
                    col_indices.append(k - 1)
                if i < nx - 1:
                    data.append(1.0 / (hx * hx))  # Voisin à droite
                    row_indices.append(k)
                    col_indices.append(k + 1)
                if j > 0:
                    data.append(2.0 / (hy * hy))  # Voisin en haut
                    row_indices.append(k)
                    col_indices.append(k - nx)
                if j < ny - 1:
                    data.append(2.0 / (hy * hy))  # Voisin en bas
                    row_indices.append(k)
                    col_indices.append(k + nx)

    # Conversion des listes en tableaux numpy
    data = np.array(data, dtype=complex)
    row_indices = np.array(row_indices)
    col_indices = np.array(col_indices)

    # Création de la matrice CSR
    matrix = csr_matrix((data, (row_indices, col_indices)), shape=(nx * ny, nx * ny))

    return matrix

def BC_dirichlet_csr(T):
    nx, ny = T.shape

    # Listes pour stocker les valeurs non nulles de la matrice
    data = []
    row_indices = []
    col_indices = []

    # Remplissage de la matrice
    for i in range(nx):
        for j in range(ny):
            k = i + j * nx  # Indice de la cellule (i, j)
            if T[i, j] == 72 or T[i,j] == 77:
                data.append(1)  # Valeur de la diagonale
                row_indices.append(k)
                col_indices.append(k)

    # Conversion des listes en tableaux numpy
    data = np.array(data, dtype=complex)
    row_indices = np.array(row_indices)
    col_indices = np.array(col_indices)

    # Création de la matrice CSR
    matrix = csr_matrix((data, (row_indices, col_indices)), shape=(nx * ny, nx * ny))

    return matrix

def BC_onde_csr(T):
    nx, ny = T.shape
    b = np.zeros((nx * ny, 1), dtype=complex)

    # Listes pour stocker les valeurs non nulles de la matrice
    data = []
    row_indices = []
    col_indices = []

    for i in range(nx):
        for j in range(ny):
            k = i + j * nx

            if T[i, j] == 8:
                b[k, 0] = -np.exp(complex(0, k0 * (ny - j))) * i * (nx - i) / (hy * hy)
                data.append(-2.0 / (hx * hx) - 2.0 / (hy * hy) + k0 * k0)  # Valeur de la diagonale
                row_indices.append(k)
                col_indices.append(k)

                if i > 0:
                    data.append(1.0 / (hx * hx))
                    row_indices.append(k)
                    col_indices.append(k - 1)  # Voisin à gauche
                if i < nx - 1:
                    data.append(1.0 / (hx * hx))
                    row_indices.append(k)
                    col_indices.append(k + 1)  # Voisin à droite
                if j > 0:
                    data.append(1.0 / (hy * hy))
                    row_indices.append(k)
                    col_indices.append(k - nx)  # Voisin en haut
                if j < ny - 1:
                    data.append(1.0 / (hy * hy))
                    row_indices.append(k)
                    col_indices.append(k + nx)  # Voisin en bas

    # Conversion des listes en tableaux numpy
    data = np.array(data, dtype=complex)
    row_indices = np.array(row_indices)
    col_indices = np.array(col_indices)

    # Création de la matrice CSR
    matrix = csr_matrix((data, (row_indices, col_indices)), shape=(nx * ny, nx * ny))
    
    # Conversion de la matrice Force en un vecteur colonne CSR
    b_csr = csr_matrix(b)
    
    return matrix, b_csr

T = np.array([  [1, 1, 1,1,1,1, 1, 0,0,0,0],
                [8, 2, 2,2,2,2, 1, 0,0,0,0],
                [8, 2, 2,2,2,2, 1, 0,0,0,0],
                [8, 2, 2,2,2,2, 1, 0,0,0,0],
                [8, 2, 2,2,2,2, 1, 0,0,0,0],
                [8, 2, 2,2,2,2, 1, 0,0,0,0],
                [8, 2, 2,2,2,2, 1, 1,1,1,1],
                [8, 2, 2,2,2,2, 2, 2,2,2,7],
                [8, 2, 2,2,2,2, 2, 2,2,2,7],
                [8, 2, 2,2,2,2, 2, 2,2,2,7],
                [8, 2, 2,2,2,2, 2, 2,2,2,7],
                [8, 2, 2,2,2,2, 2, 2,2,2,7],
                [8, 2, 2,2,2,2, 2, 2,2,2,7],
                [8, 2, 2,2,2,2, 2, 2,2,2,7],
                [8, 2, 2,2,2,2, 2, 2,2,2,7],
                [8, 2, 2,2,2,2, 2, 2,2,2,7],
                [8, 2, 2,2,2,2, 2, 2,2,2,7],
                [1, 1, 1,1,1,1, 1, 1,1,1,1],
                ])

T = mapp.reduction("images/exemple7.png")
 
# anat.VisualizeMatrix(T)
print(T)

height, width = T.shape
free_nodes = sorted(list(filter( lambda x: x != None, [ i + height * j if T[i, j] != 0 else None for i in range(height) for j in range(width)])))
all_nodes = np.arange(width * height)
fixed_nodes = np.setdiff1d(all_nodes, free_nodes)

#lecture matrice
# T =np.array([[7, 2],
#              [7, 2],
#              [7, 2, 2],])


# A,b=BC_onde(T)
# A += laplacien(T)+identite(T)+BC_dirichlet(T)+BC_robin(T)

A,b= BC_onde_csr(T)
A += identite_csr(T)+laplacien_csr(T)+BC_robin_csr(T)+BC_dirichlet_csr(T)
b+= force_test(T)

# Vérification de la symétrie
# print(A)
# is_symmetric = np.array_equal(BC_robin(T), BC_robin(T).T)
# print("La matrice est-elle symétrique ?", is_symmetric)

# # Vérification de la définie positivité
# eigenvalues = np.linalg.eigvals(A)
# is_positive_definite = np.all(eigenvalues > 0)
# print("La matrice est-elle définie positive ?", is_positive_definite)



# rows_to_remove = fixed_nodes  # Indices of rows to remove (2nd and 4th rows)
# cols_to_remove = fixed_nodes  # Indices of columns to remove (1st and 3rd columns)

# Remove specified rows
#A_without_rows = np.delete(A, rows_to_remove, axis=0)
#A_without_rows_cols = np.delete(A_without_rows, cols_to_remove, axis=1)
#A = A_without_rows_cols


A = A[free_nodes, :]
A = A[ : , free_nodes]

#A = A[free_nodes, :]

# b = np.delete(b, rows_to_remove, axis=0)

b = b[free_nodes, :]
# print(type(b))

# Conversion de la matrice CSR en dense
A_dense = A.toarray()

# Calcul du déterminant de la matrice dense
det_A_dense = np.linalg.det(A_dense)

# Vérification si la matrice est singulière
if det_A_dense == 0:
    print("La matrice est singulière.")
else:
    print("La matrice n'est pas singulière.")

# Calcul du déterminant


A_inverse = inv(A)


# print(np.linalg.inv(A) @ b)

a = 1

if a == 1:
    v = np.zeros((height * width, 1), dtype=complex)
    # v[free_nodes] = np.linalg.inv(A) @ b
    v[free_nodes]=A_inverse.dot(b).toarray()
    nx, ny = T.shape
    u = np.reshape(v, (ny, nx))
    w = np.reshape(v, (nx, ny)) 

    print(anat.erreurs(u))

    # Extraire les parties réelles et imaginaires de la solution
    u_real = np.real(u)
    u_imag = np.imag(u)

    # Définir les dimensions de votre grille 2D
    nx, ny = u_real.shape

    # Créer les grilles X et Y de manière appropriée
    X, Y = np.meshgrid(np.arange(ny), np.arange(nx))

    # Créer la figure
    fig = plt.figure(figsize=(12, 6))

    # Afficher la solution u en 3D (partie réelle)
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, Y, u_real, cmap='viridis', edgecolor='none')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('u_real')
    ax1.set_title('Partie réelle de la solution u')

    # Afficher la solution u en 3D (partie imaginaire)
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, Y, u_imag, cmap='viridis', edgecolor='none')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('u_imag')
    ax2.set_title('Partie imaginaire de la solution u')

    # plt.tight_layout()
    # plt.show()



    fig2, ax3, ax4 = anat.fonction(u)

    # Ajouter des titres aux graphiques de la première figure
    ax1.set_title("Partie Réelle calcué")
    ax2.set_title("Partie Imaginaire calculé")

    # Ajouter des titres aux graphiques de la deuxième figure
    ax3.set_title("partie réelle analityque")
    ax4.set_title("partie imaginaire analityque")

    plt.tight_layout()
    plt.show()