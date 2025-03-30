import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import Mapping as mapp
import numpy as np
from scipy.sparse.linalg import spsolve
import fonction_analytique as anat
import time
from matrice_operateurs_wholeLaplace import *
import matrice_operateurs_basicLaplace as basic
import affichage as aff


"""
Convention of domain:
1 : Dirichlet
2 : Domain
7X: Neumann out
8X: Onde in

"""

########## Parameters ###########
value = aff.affichage() #Lx,Ly,bc_side,solution,domain_shape,domain_shape,force,fig_anat,multiplier_4_increasing_resolution
Lx = value[0]  # from 10 to 200
Ly = value[1] # from 10 to 200
display = value[2]
bc_side = value[3] # 0 -> "dirichlet" ou 1 -> "neumann"
solution = value[4]   # "basic" ou "complet"
domain_shape = value[5]
force = value[6] #"test" ou "irregulier" ou "0"
fig_anat= value[7] #"comparaison" ou "reacteur"



multiplier_4_increasing_resolution = value[7] # from 1 to 10 

if force == "Test":
   origin = "rect"
   img_path=None
elif force == "Irregular":
   origin = "image"
   img_path="Irregulier.png"
elif force == "0":
   origin = "image"
   img_path="Irregulier.png"

########## Timer matrix starter ###########
time_matrix_builder_start = time.time()

########## Parameters ###########
""" we can choose: rect - img - array """
T = anat.setting_matrix_domain(origin = origin,image_path =img_path, nodes_x = Lx * multiplier_4_increasing_resolution + 1, nodes_y = Ly * multiplier_4_increasing_resolution + 1)

nodes_y, nodes_x = T.shape


metric = anat.Metric(Lx, Ly, nodes_x, nodes_y)

free_nodes = list(filter( lambda x: x != None, [ i + nodes_y * j if T[i, j] != 0 else None for i in range(nodes_y) for j in range(nodes_x)]))
all_nodes = np.arange(nodes_x * nodes_y) 
fixed_nodes = np.setdiff1d(all_nodes, free_nodes)



####### Defining functions of boundary conditions ######

if force == "Test":
   f = lambda x,y: -2*np.exp(complex(0, k * x))
   g = lambda x, y: y * (metric.Ly - y)
   f_solution = None

elif force == "Irregulier":
   f = lambda x,y:(y+Ly-1)/(5*(Lx-1)*(Ly-1))*\
            (y*y-(Ly-1)*y*(x/(5*(Lx-1))+1)+(Ly-1)*(Ly-1)*x/(5*(Lx-1)))\
            *np.exp(x*np.complex(-(Ly-1)/(5*(Lx-1))*(y+(Ly-1))/(y*y+(Ly-1)*y),k))\
            if y != 0 else 0
   
   g = lambda x,y:(y+Ly-1)/(5*(Lx-1)*(Ly-1))*\
            (y*y-(Ly-1)*y*(x/(5*(Lx-1))+1)+(Ly-1)*(Ly-1)*x/(5*(Lx-1)))
   
   f_solution = "Irregulier"
   
elif force == "0":
   f = lambda x, y: 0
   g = lambda x, y: 1
   f_solution = "Irregulier"

else :
   #f = lambda x, y: ((M0**2 * k0**2 + 2 * M0 * k0**2) * y * (metric.Ly - y) - 2 ) * np.exp(complex(0, k0 * x))
   #f = lambda x, y: (2*k0**2*M0 + M0**2*k0**2) * np.exp(complex(0, k0 * x))
   f = lambda x, y: 0
   g = lambda x, y: 1
   f_solution = "Irregulier"
   

####### Building the matrixes ########

if solution == "Complet":

   A, b  = BC_onde_csr(T, g = g, metric = metric)
   A2,b2 = BC_up_down_csr(T, f, bc_side = bc_side ,metric = metric)
   A    += identite_csr(T, scalar = k0 ** 2, metric = metric) + laplacien_csr(T, metric = metric)
   A    += Dx2_csr(T, scalar = -1 * M0**2, metric = metric) + Dx1_csr(T, scalar = complex( 0, -2 * k0 * M0), metric = metric )
   A    += BC_neumann_csr(T, metric = metric) 
   b    += force_test(T, f = f_solution, metric = metric)
   A += A2
   b += b2

elif solution == "Basic":
   A, b  = basic.BC_onde_csr(T, g = g, metric = metric)
   A2,b2 = basic.BC_dirichlet_csr(T, f,metric = metric)
   A    += basic.identite_csr(T, scalar = k0 ** 2, metric = metric) + laplacien_csr(T, metric = metric)
   # A    += basic.Dx2_csr(T, scalar = 0, metric = metric) + Dx1_csr(T, scalar = complex( 0, 0), metric = metric )
   A    += basic.BC_neumann_csr(T, metric = metric) 
   b    += basic.force_test(T, f = f_solution, metric = metric)
   A += A2
   b += b2
###### we are erasing the fixed nodes ######
A = A[free_nodes, :]
A = A[ : , free_nodes]
b = b[free_nodes]



########## Timer end ###########
time_matrix_builder_end = time.time()



##### solving the system #####

time_solving_start = time.time()

v = np.zeros((nodes_y * nodes_x), dtype=complex)
v[free_nodes] = spsolve(A, b)

time_solving_end = time.time()



########## Timer end ###########
time_matrix_builder = time_matrix_builder_end - time_matrix_builder_start
time_matrix_builder = round(time_matrix_builder, 2)
time_solving = time_solving_end - time_solving_start
time_solving = round(time_solving, 2)



""" 
Be careful with the reshape. I think there are sometimes
that the reshape is not working as expected.
I think it works row wise, so we have to transpose the matrix
"""

####### Solution #######

u = np.reshape(v, (nodes_x, nodes_y)).T
real_error, imaginary_error = anat.erreurs(u, f = f_solution, metric = metric)

text = "Real errors: {:.2f}% | Imaginary errors: {:.2f}%\n".format(real_error, imaginary_error)
text += "Time building matrix: {:.2f} s | Time solving system: {:.2f} s\n".format(time_matrix_builder, time_solving)
text += "Total time: {:.2f} s | h: {:.2f}".format(time_matrix_builder + time_solving, metric.h)

print(text)

if display:
   if fig_anat == "Reacteur":
      # #### Créer la figure numerique #####
      # fig = plt.figure(figsize=(12, 6))
      # ax1, fig = anat.plot_numerical_real(u, fig, metric)
      # ax2, fig = anat.plot_numerical_img(u, fig, metric)
      # # Ajouter une zone de texte en dessous de la figure
      # fig.text(0.5, 0.05, text, ha='center', fontsize=12)
      # #### Créer la figure analytique #####
      # fig2, ax3, ax4 = anat.plot_analytical_solution(u, f = f_solution, metric = metric) 
      #### Créer la figure numerique #####
      fig = plt.figure(figsize=(12, 6))
      ax1, fig = anat.plot_numerical_real(u, fig, metric)
      ax2, fig = anat.plot_numerical_img(u, fig, metric)
      # Ajouter une zone de texte en dessous de la figure
      fig.text(0.5, 0.05, text, ha='center', fontsize=12)

   elif fig_anat =="Comparaison":
      # fig2 = plt.figure(figsize=(12, 6))
      fig2,ax1,ax2=anat.plot_analytical_solution(u, f = f_solution, metric = metric,a=5) 
      fig2.text(0.5, 0.035, text, ha='center', fontsize=12)
   plt.show()