import numpy as np
import matplotlib.pyplot as plt
from parameters import *
import Mapping as mapp
from os import path
import numpy as np


def setting_matrix_domain ( origin = None, nodes_x = None, nodes_y = None, image_path = None):
    
    if origin is "square" or origin is "rectangle" or origin is "carre" or origin is "rectangulaire" \
        or "rect" in origin or "car" in origin or "squa" in origin or "rec" in origin \
        or "carre" in origin or "rectangle" in origin or "square" in origin \
        or "rectangulaire" in origin or "rect" in origin or "car" in origin \
        or "squa" in origin or "rec" in origin:
        
        if nodes_x is None:
            nodes_x = 60
        if nodes_y is None:
            nodes_y = 20       
        
        T = np.ones((int(np.ceil(nodes_y)), int(np.ceil(nodes_x)))) * 2
        
        nodes_y, nodes_x = T.shape


        
        T[0, :] = 72
        T[nodes_y - 1, :] = 77
        T[ : , nodes_x- 1 ] = 8 #
        T[ : , 0] = 7 #


        T[0,0 ] = 91
        T[nodes_y - 1, 0] = 96
        # T[0, nodes_x - 1] = 13
        # T[nodes_y - 1, nodes_x - 1] = 18
        
    
    elif "img" in origin or "image" in origin:
        
        if image_path is None:
            raise Exception("Please provide a valid path to the image")
        T = mapp.reduction(path.join("images", image_path))
        T = np.flipud(T)
        
        
    elif "array" in origin or "matrice" in origin:

        T = np.array([  [ 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                        [74, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0],
                        [74, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0],
                        [74, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0],
                        [74, 2, 2, 2, 2, 2, 1, 1, 1, 1, 0],
                        [74, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0],
                        [74, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1],
                        [74, 2, 2, 2, 2, 2, 2, 2, 2, 2,85],
                        [74, 2, 2, 2, 2, 2, 2, 2, 2, 2,85],
                        [74, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1],
                        [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        ])
    else: 
        raise Exception("Please provide a valid origin")
    return T
        
        

class Metric:
    
    def __init__(self, Lx, Ly, nodes_x, nodes_y):
        self.Lx = Lx
        self.Ly = Ly
        self.nodes_x = nodes_x
        self.nodes_y = nodes_y
        self.dx = Lx / (nodes_x - 1)    #remember that we have nodes_x - 1 intervals
        self.dy = Ly / (nodes_y - 1)    #remember that we have nodes_y - 1 intervals

    @property
    def hx(self):
        return self.dx
    @property
    def hy(self):
        return self.dy
    @property
    def h(self):
        return self.dx
    def __str__(self):
        return "Lx = {}, Ly = {}, nodes_x = {}, nodes_y = {}, dx = {}, dy = {}".format(self.Lx, self.Ly, self.nodes_x, self.nodes_y, self.dx, self.dy)



def plot_analytical_solution(u, a=1, f = None, metric = None):
        
    nodes_y, nodes_x = u.shape
    
    if f == None:
        f = lambda x, y: y * (Ly - y) * np.exp(complex(0, k * (x)))

    elif f == "Irregulier":
        f = lambda x,y:(y+Ly-1)/(10*(Lx-1)*(Ly-1))*\
            (y*y-(Ly-1)*y*(x/(10*(Lx-1))+1)+(Ly-1)*(Ly-1)*x/(10*(Lx-1)))\
            *np.exp(x*np.complex(-(Ly-1)/(10*(Lx-1))*(y+(Ly-1))/(y*y+(Ly-1)*y),k))\
            if y != 0 else 0

    Z = np.zeros((nodes_y, nodes_x), dtype=complex)
    for i in range(nodes_y):
        for j in range(nodes_x):
            x = j * metric.hx
            y = metric.Ly - i * metric.hy
            
            Z[i,j]= f(x, y)


    # Obtenez la partie réelle et la partie imaginaire de Z
    Z_real = np.real(Z)
    Z_imag = np.imag(Z)
    

    # How to use np.linspace    
    """ 
    # Generate an array of evenly spaced values between start and stop
    x = np.linspace(start, stop, num)

    # Example usage
    start = 0
    stop = 10
    num = 5
    x = np.linspace(start, stop, num)
    print(x) """

    X, Y = np.meshgrid(np.linspace(0, metric.Lx, nodes_x), np.linspace(0, metric.Ly, nodes_y))

    if a==2:
        # Créez une figure 3D
        fig1 = plt.figure(figsize=(12, 5))

        # Graphique de la partie réelle
        ax1 = fig1.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(X, Y, Z_real, cmap='coolwarm', edgecolor='none')
        ax1.set_title("Partie Réelle")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z_real")
        fig1.colorbar(surf1, ax=ax1)

        # Graphique de la partie imaginaire
        ax2 = fig1.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(X, Y, Z_imag, cmap='coolwarm', edgecolor='none')
        ax2.set_title("Partie Imaginaire")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z_imag")
        fig1.colorbar(surf2, ax=ax2)
        # Affichez la figure
        plt.show()
    
    elif a ==3:
        return Z_real,Z_imag
    
    elif a == 4:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Affichez la partie réelle avec une colormap coolwarm
        im1 = ax1.imshow(Z_real, cmap='coolwarm', origin='lower')
        ax1.set_title("Partie Réelle")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        fig.colorbar(im1, ax=ax1)

        # Affichez la partie imaginaire avec une colormap coolwarm
        im2 = ax2.imshow(Z_imag, cmap='coolwarm', origin='lower')
        ax2.set_title("Partie Imaginaire")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        fig.colorbar(im2, ax=ax2)

        # Affichez la figure
        plt.show()
    
    elif a == 5:

        # Créez une figure 3D
        fig1 = plt.figure(figsize=(14, 8.5))

        # Graphique de la partie réelle
        ax1 = fig1.add_subplot(221, projection='3d')
        surf1 = ax1.plot_surface(X, Y, Z_real, cmap='coolwarm', edgecolor='none')
        ax1.set_title("Partie Réelle analytique")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Pression")
        fig1.colorbar(surf1, ax=ax1)

        # Graphique de la partie réelle de u
        ax2 = fig1.add_subplot(223, projection='3d')
        surf2 = ax2.plot_surface(X, Y, u.real, cmap='coolwarm', edgecolor='none')
        ax2.set_title("Partie Réelle calculé")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Pression")
        fig1.colorbar(surf2, ax=ax2)

        # Graphique de la partie imaginaire
        ax2 = fig1.add_subplot(222, projection='3d')
        surf2 = ax2.plot_surface(X, Y, Z_imag, cmap='coolwarm', edgecolor='none')
        ax2.set_title("Partie Imaginaire analytique")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Pression")
        fig1.colorbar(surf2, ax=ax2)

        # Graphique de la partie réelle de u
        ax2 = fig1.add_subplot(224, projection='3d')
        surf2 = ax2.plot_surface(X, Y, u.imag, cmap='coolwarm', edgecolor='none')
        ax2.set_title("Partie imaginaire calculé")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Pression")
        fig1.colorbar(surf2, ax=ax2)

        return fig1,ax1,ax2


    else :
        # Créez une figure 3D
        fig1 = plt.figure(figsize=(12, 5))

        # Graphique de la partie réelle
        ax1 = fig1.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(X, Y, Z_real, cmap='coolwarm', edgecolor='none')
        ax1.set_title("Partie Réelle")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z_real")
        fig1.colorbar(surf1, ax=ax1)

        # Graphique de la partie imaginaire
        ax2 = fig1.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(X, Y, Z_imag, cmap='coolwarm', edgecolor='none')
        ax2.set_title("Partie Imaginaire")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z_imag")
        fig1.colorbar(surf2, ax=ax2)
        # Ajouter des titres aux graphiques de la deuxième figure
        ax1.set_title("partie réelle analityque")
        ax2.set_title("partie imaginaire analityque")
        return fig1, ax1, ax2


def plot_numerical_real(u, fig, metric : Metric = None):
    # Définir les dimensions de votre grille 2D
    nodes_y, nodes_x = u.shape

    # Créer les grilles X et Y de manière appropriée
    X, Y = np.meshgrid(np.linspace(0, metric.Lx, nodes_x), np.linspace(0, metric.Ly, nodes_y))

    # Afficher la solution u en 3D (partie réelle)
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, Y, np.real(u), cmap='viridis', edgecolor='none')
    fig.colorbar(surf1, ax=ax1)
    
    #ax1.set_box_aspect([np.ptp(X), np.ptp(Y), min(np.ptp(X), np.ptp(Y))])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('u_real')
    ax1.set_title("Partie Réelle calcué")
    
    return ax1, fig


def plot_numerical_img(u, fig, metric = None):
    # Définir les dimensions de votre grille 2D
    nodes_y, nodes_x = u.shape

    # Créer les grilles X et Y de manière appropriée
    X, Y = np.meshgrid(np.linspace(0, metric.Lx, nodes_x), np.linspace(0, metric.Ly, nodes_y))

    # Afficher la solution u en 3D (partie imaginaire)
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, Y, np.imag(u), cmap='viridis', edgecolor='none')
    fig.colorbar(surf2, ax=ax2)
    
    #ax2.set_box_aspect([np.ptp(X), np.ptp(Y), min(np.ptp(X), np.ptp(Y))])
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('u_imag')
    ax2.set_title("Partie Imaginaire calculé")
    
    return ax2, fig



def erreurs(u, f = None, metric : Metric = None):
    Z_real,Z_imag = plot_analytical_solution(u, a = 3, f = f, metric = metric)
    nx,ny = u.shape
    
    hx = metric.hx
    hy = metric.hy
    
    # errors_R = np.linalg.norm(np.real(u) - Z_real) / np.linalg.norm(Z_real) * 100
    # errors_I = np.linalg.norm(np.imag(u) - Z_imag) / np.linalg.norm(Z_imag) * 100

    Real_errors = 0
    imag_errors = 0
    real_anat = 0
    imag_anat = 0

    for j in range(ny):
        for i in range(nx):
            Real_errors += abs(Z_real[i,j]-np.real(u[i,j]))**2*hx*hy
            imag_errors +=abs(Z_imag[i,j]-np.imag(u[i,j]))**2*hx*hy
            real_anat += Z_real[i,j]**2*hx*hy
            imag_anat += Z_imag[i,j]**2*hx*hy
    
    epsilon = 1e-10
    if abs(real_anat) <= epsilon:
        Real_errors = -1
    else:
        Real_errors = np.sqrt(Real_errors)/np.sqrt(real_anat)*100
        
    if abs(imag_anat) <= epsilon:
        imag_errors = -1
    else:
        imag_errors = np.sqrt(imag_errors)/np.sqrt(imag_anat)*100
    
    return Real_errors,imag_errors



def erreur_sur_domaine(u, erreur_real, erreur_img, f = None, metric : Metric = None):
    Z_real,Z_imag = plot_analytical_solution(u, a = 3, f = f, metric = metric)
    
    u_diff_real = abs(Z_real - np.real(u)) 
    u_diff_imag = abs(Z_imag - np.imag(u)) 

    return u_diff_real, u_diff_imag



def VisualizeSparseMatrix (sparse_matrix, show=True, title = None):
    # Extract real and imaginary parts
    real_Dx = np.real(sparse_matrix.toarray())
    imag_Dx = np.imag(sparse_matrix.toarray())

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    if title:
        fig.suptitle(title)

    # Plot real part
    cax0 = axs[0].imshow(real_Dx, cmap='viridis')
    axs[0].set_title('Real part of Dx Matrix')
    fig.colorbar(cax0, ax=axs[0])

    # Plot imaginary part
    cax1 = axs[1].imshow(imag_Dx, cmap='viridis')
    axs[1].set_title('Imaginary part of Dx Matrix')
    fig.colorbar(cax1, ax=axs[1])
    
    if show:
        plt.show()
    
    
def VisualizeMatrix (matrix, minmax = None):
    # Extract real and imaginary parts
    real_Dx = np.real(matrix)
    imag_Dx = np.imag(matrix)
    
    if minmax:
        minimum = minmax[0]
        maximum = minmax[1]
        
        # Create subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot real part
        cax0 = axs[0].imshow(real_Dx, cmap='viridis', vmin=minimum, vmax=maximum)
        axs[0].set_title('Real part of Dx Matrix')
        fig.colorbar(cax0, ax=axs[0])

        # Plot imaginary part
        cax1 = axs[1].imshow(imag_Dx, cmap='viridis', vmin=minimum, vmax=maximum)
        axs[1].set_title('Imaginary part of Dx Matrix')
        fig.colorbar(cax1, ax=axs[1])

        plt.show()
    
    else:
        # Create subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot real part
        cax0 = axs[0].imshow(real_Dx, cmap='viridis')
        axs[0].set_title('Real part of Dx Matrix')
        fig.colorbar(cax0, ax=axs[0])

        # Plot imaginary part
        cax1 = axs[1].imshow(imag_Dx, cmap='viridis')
        axs[1].set_title('Imaginary part of Dx Matrix')
        fig.colorbar(cax1, ax=axs[1])

        plt.show()
        
        
        


        