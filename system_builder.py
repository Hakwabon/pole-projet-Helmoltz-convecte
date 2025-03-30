import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import Mapping as mapp
import numpy as np
from scipy.sparse.linalg import spsolve
import fonction_analytique as anat
import time
import matrice_operateurs_wholeLaplace as mtwhole
import matrice_operateurs_basicLaplace as mtbasic
import parameters as pr
import os

"""
Convention of domain:
1 : Dirichlet
2 : Domain
7X: Neumann out
8X: Onde in

"""

class SystemBuilder:
    def __init__(self, Lx, Ly, bc_side, operators_used, domain_shape, multiplier_4_increasing_resolution, f = None, g = None, f_solution = None, display = False, save = False, final_setup = False, image_path = None):
        self.Lx = Lx
        self.Ly = Ly
        self.__bc_side = bc_side
        self.__operators_used = operators_used
        self.display = display
        self.__domain_shape = domain_shape
        self.__multiplier_4_increasing_resolution = multiplier_4_increasing_resolution
        self.save = save
        self.final_setup = final_setup
        self.__T = anat.setting_matrix_domain(origin = self.domain_shape, nodes_x = Lx * multiplier_4_increasing_resolution + 1, nodes_y = Ly * multiplier_4_increasing_resolution + 1, image_path = image_path)
        
        anat.VisualizeMatrix(self.__T)
        
        self.set_functions(f, g, f_solution)
        
        self.nodes_y, self.nodes_x = self.T.shape
        
        self.metric = anat.Metric(Lx, Ly, self.nodes_x, self.nodes_y)
        self.__free_nodes = list(filter( lambda x: x != None, [ i + self.nodes_y * j if self.T[i, j] != 0 else None for i in range(self.nodes_y) for j in range(self.nodes_x)]))
        self.__all_nodes = np.arange(self.nodes_x * self.nodes_y) 
        self.__fixed_nodes = np.setdiff1d(self.__all_nodes, self.__free_nodes)
        
        self.__time_matrix_build = 0
        self.__time_solve_system = 0
        
        self.__A = None
        self.__b = None
        self.__u = None
        self.__real_error = None
        self.__imaginary_error = None
        self.__u_diff_imag = None
        self.__u_diff_real = None
    
    
    @property
    def T (self):
        return self.__T
    
    @property
    def A (self):
        if self.__A is None:
            raise ValueError("Matrix not built yet")
        return self.__A
    
    @property
    def b (self):
        if self.__b is None:
            raise TypeError("Matrix not built yet")
        return self.__b
    
    @property
    def bc_side (self):
        options = (
            "dirichlet",
            "robin",
            "neumann"
        )
        
        if not type(self.__bc_side) is int and not type(self.__bc_side) is float:
            raise TypeError("bc_side must be an integer")
        
        if self.__bc_side not in range(len(options)):
            raise ValueError("Invalid bc_side")
        
        return options[ self.__bc_side ]

    @property
    def operators_used (self):
        options = (
            "basic",
            "complet"
        )
        
        if not type(self.__operators_used) is int and not type(self.__operators_used) is float:
            raise TypeError("operators_used must be an integer")
        
        if self.__operators_used not in range(len(options)):
            raise ValueError("Invalid operators_used")
        
        return options[ self.__operators_used ]
    
    @property
    def domain_shape (self):
        options = (
            "rect",
            "img",
            "array",
            "irregular"
        )
        
        if not type(self.__domain_shape) is int and not type(self.__domain_shape) is float:
            raise TypeError("domain_shape must be an integer")
        
        if self.__domain_shape not in range(len(options)):
            raise ValueError("Invalid domain_shape")
        
        return options[ self.__domain_shape ]
    
    @property
    def time_matrix_build (self):
        return self.__time_matrix_build
    
    @property
    def time_solve_system (self):
        return self.__time_solve_system
    
    @property
    def u (self):
        if self.__u is None:
            raise ValueError("System not solved yet")
        return self.__u
    
    @property
    def real_error (self):
        if self.__real_error is None:
            raise ValueError("System not solved yet")
        return self.__real_error
    
    @property
    def imaginary_error (self):
        if self.__imaginary_error is None:
            raise ValueError("System not solved yet")
        return self.__imaginary_error
    
    
    def set_functions(self, f, g, f_solution):
        
        if self.final_setup:
            self.__f = lambda x, y: 0
            self.__g = (lambda x, y: y * (self.metric.Ly - y)) if g is None else g
            self.__f_solution = lambda x, y: 0
        
        else:
        
            if self.domain_shape == "rect":
                    
                    if self.operators_used == "basic":
                        
                        if self.bc_side == "dirichlet":
                            self.__f =  lambda x, y: -2 * np.exp( complex(0, pr.k * x))
                            self.__g = lambda x, y: y * (self.metric.Ly - y)
                            self.__f_solution = lambda x, y: y * (self.Ly - y) * np.exp(complex(0, pr.k * x))
                            
                        elif self.bc_side == "neumann":
                            eq1 = lambda k, x, Ly, y: -(16 * np.exp(1j * k * x) * np.pi**2 * np.cos((4 * np.pi * y) / Ly)) / Ly**2
                            self.__f = lambda x, y: eq1(pr.k0, x, self.Ly, y)
                            
                            self.__g = lambda x, y: np.cos(4 * np.pi * y / self.Ly)
                            
                            self.__f_solution = lambda x, y: np.cos(4 * np.pi * y / self.Ly) * np.exp(1j * pr.k0 * x)
                            
                        elif self.bc_side == "robin":
                            #raise ValueError("Basic equation. Neumann not implemented yet")
                            
                            equation_f_sol = lambda Ly, k, M, eta, x, y: (
                                np.exp((1j * (k * Ly * x + np.pi * y)) / Ly) * (
                                    np.exp(x * (-(1j * k * (-1 + M)) / M + (np.sqrt(k) * np.sqrt(np.pi)) / (M * np.sqrt(Ly * eta)))) + 
                                    (np.exp(x * (-(1j * k * (-1 + M)) / M - (np.sqrt(k) * np.sqrt(np.pi)) / (M * np.sqrt(Ly * eta)))) * 
                                    (np.sqrt(np.pi) - 1j * np.sqrt(k) * (-1 + M) * np.sqrt(Ly * eta))) / 
                                    (np.sqrt(np.pi) + 1j * np.sqrt(k) * (-1 + M) * np.sqrt(Ly * eta))
                                )
                            )
                            
                            self.__f_solution = lambda x, y:  equation_f_sol(self.Ly, pr.k0, pr.M0, pr.eta, x, y)
                            
                            
                            equation_g = lambda Ly, k, M, eta, x, y: (
                                np.exp((1j * np.pi * y) / Ly) * (
                                    np.exp(x * (-(1j * k * (-1 + M)) / M + (np.sqrt(k) * np.sqrt(np.pi)) / (M * np.sqrt(Ly * eta)))) + 
                                    (np.exp(x * (-(1j * k * (-1 + M)) / M - (np.sqrt(k) * np.sqrt(np.pi)) / (M * np.sqrt(Ly * eta)))) * 
                                    (np.sqrt(np.pi) - 1j * np.sqrt(k) * (-1 + M) * np.sqrt(Ly * eta))) / 
                                    (np.sqrt(np.pi) + 1j * np.sqrt(k) * (-1 + M) * np.sqrt(Ly * eta))
                                )
                            )
                            
                            self.__g = lambda x, y: equation_g(self.Ly, pr.k0, pr.M0, pr.eta, x, y)
                            
                            
                            equation_f = lambda Ly, k, M, eta, x, y: (
                                (1 / (Ly**2 * M**2 * eta)) * 
                                np.exp((1j * (k * Ly * x + np.pi * y)) / Ly) * (
                                    np.exp(x * (-(1j * k * (-1 + M)) / M + (np.sqrt(k) * np.sqrt(np.pi)) / (M * np.sqrt(Ly * eta)))) * 
                                    (k * Ly * np.pi + k**2 * Ly**2 * (-1 + M**2) * eta - M**2 * np.pi**2 * eta + 
                                    2j * k**(3/2) * Ly * np.sqrt(np.pi) * np.sqrt(Ly * eta)) + 
                                    np.exp(x * (-(1j * k * (-1 + M)) / M - (np.sqrt(k) * np.sqrt(np.pi)) / (M * np.sqrt(Ly * eta)))) * 
                                    (k * Ly * np.pi**(3/2) + k**2 * Ly**2 * (-1 + M)**2 * np.sqrt(np.pi) * eta - 
                                    M**2 * np.pi**(5/2) * eta - 
                                    1j * k**(3/2) * Ly * (1 + M) * np.pi * np.sqrt(Ly * eta) + 
                                    1j * np.sqrt(k) * (-1 + M) * M**2 * np.pi**2 * eta * np.sqrt(Ly * eta) - 
                                    1j * k**(5/2) * Ly * (-1 + M)**2 * (1 + M) * (Ly * eta)**(3/2))
                                ) / (np.sqrt(np.pi) + 1j * np.sqrt(k) * (-1 + M) * np.sqrt(Ly * eta))
                            )
                            
                            self.__f = lambda x, y: equation_f(self.Ly, pr.k0, pr.M0, pr.eta, x, y)
                            
                            
                            
                            
                        
                    elif self.operators_used == "complet":
                        
                        if self.bc_side == "dirichlet":
                            self.__f = lambda x, y: ((pr.M0**2 * pr.k0**2 + 2 * pr.M0 * pr.k0**2) * y * (self.metric.Ly - y) - 2 ) * np.exp(complex(0, pr.k0 * x))
                            self.__g = lambda x, y: y * (self.metric.Ly - y)
                            self.__f_solution = lambda x, y: y * (self.Ly - y) * np.exp(complex(0, pr.k * x))
                            
                            
                        elif self.bc_side == "neumann":
                            eq1 = lambda k, x, Ly, M, y: (np.exp(1j * k * x) * (k**2 * Ly**2 * M * (2 + M) - 16 * np.pi**2) * np.cos((4 * np.pi * y) / Ly)) / Ly**2        
                            self.__f = lambda x, y: eq1(pr.k0, x, self.Ly, pr.M0, y)
                            
                            self.__g = lambda x, y: np.cos(4 * np.pi * y / self.Ly)
                            
                            self.__f_solution = lambda x, y: np.cos(4 * np.pi * y / self.Ly) * np.exp(1j * pr.k0 * x)
                            
                            
                        elif self.bc_side == "robin":
                            # raise ValueError("Complete equation. Neumann not implemented yet")
                            
                            eq_f_sol = lambda k, x, M, y, eta, Ly: np.exp(1j * k * x) * (1 + x**2) * (
                                1 + (
                                    1j * (k**2 * (1 + M)**2 + k**2 * x**2 + 2 * k * M * x * (-2j + k * x) + 
                                    M**2 * (-2 - 4j * k * x + k**2 * x**2)) * y * eta
                                ) / (k * (1 + x**2))
                                - (
                                    1j * (k**2 * (1 + M)**2 + k**2 * x**2 + 2 * k * M * x * (-2j + k * x) + 
                                    M**2 * (-2 - 4j * k * x + k**2 * x**2)) * y**2 * eta
                                ) / (k * Ly * (1 + x**2))
                            )
                            self.__f_solution = lambda x,y: eq_f_sol(pr.k0, x, pr.M0, y, pr.eta, self.Ly)
                            
                            
                            eq_g = lambda k, Ly, x, M, y, eta: (
                                (1j * k**2 * (1 + M)**2 * (1 + x**2) * (Ly - y) * y * eta
                                - 4 * k * M * (1 + M) * x * y**2 * eta
                                + 2j * M**2 * y * (-Ly + y) * eta
                                + k * Ly * (1 + x**2 + 4 * M * (1 + M) * x * y * eta))
                                / (k * Ly)
                            )
                            
                            self.__g = lambda x, y: eq_g(pr.k0, self.Ly, x, pr.M0, y, pr.eta)
                            
                            
                            expr_f = lambda k, Ly, x, M, y, eta: (1 / (k * Ly * (1 + x**2))) * np.exp(1j * k * x) * (
                                -2j * (1 + x**2) * (-2 * M**2 - 4j * k * M * (1 + M) * x + k**2 * (1 + M)**2 * (1 + x**2)) * eta
                                + k**2 * (1 + x**2) * (
                                    k * Ly * (1 + x**2)
                                    + 1j * Ly * (-2 * M**2 - 4j * k * M * (1 + M) * x + k**2 * (1 + M)**2 * (1 + x**2)) * y * eta
                                    - 1j * (-2 * M**2 - 4j * k * M * (1 + M) * x + k**2 * (1 + M)**2 * (1 + x**2)) * y**2 * eta
                                )
                                - 2j * k * M * (2 * x + 1j * k * (1 + x**2)) * (
                                    k * Ly * (1 + x**2)
                                    + 1j * Ly * (-2 * M**2 - 4j * k * M * (1 + M) * x + k**2 * (1 + M)**2 * (1 + x**2)) * y * eta
                                    - 1j * (-2 * M**2 - 4j * k * M * (1 + M) * x + k**2 * (1 + M)**2 * (1 + x**2)) * y**2 * eta
                                )
                                + (1 - M**2) * (2 + 4j * k * x - k**2 * (1 + x**2)) * (
                                    k * Ly * (1 + x**2)
                                    + 1j * Ly * (-2 * M**2 - 4j * k * M * (1 + M) * x + k**2 * (1 + M)**2 * (1 + x**2)) * y * eta
                                    - 1j * (-2 * M**2 - 4j * k * M * (1 + M) * x + k**2 * (1 + M)**2 * (1 + x**2)) * y**2 * eta
                                )
                            )
                            
                            self.__f = lambda x,y: expr_f(pr.k0, self.Ly, x, pr.M0, y, pr.eta)
                            
                        
            elif self.domain_shape == "img":
                eq_f = lambda Lx, Ly, m, y, k, x, M: (1 / (Lx**4 * m**4 * y**4)) * np.exp(1j * k * x - (Ly * x) / (Lx * m * y)) * (
                    Lx**3 * m**3 * y**5 * (6 + k**2 * M * (2 + M) * y**2) +
                    Lx**2 * Ly * m**2 * y**4 * (2 * x + 4j * k * (-1 + M + M**2) * y**2 - k**2 * M * (2 + M) * x * y**2) +
                    Ly**5 * (x**3 - (-1 + M**2) * x * y**2) +
                    Lx * Ly**4 * m * y * (-3 * x**2 + 3 * (-1 + M**2) * y**2 + 2j * k * (-1 + M + M**2) * x * y**2) -
                    Lx * Ly**2 * m * y**3 * (x**2 + (k**2 * Lx**2 * m**2 * M * (2 + M) + 3 * (-1 + M**2)) * y**2 + 2j * k * (-1 + M + M**2) * x * y**2) +
                    Ly**3 * (-x**3 * y**2 - 4j * k * Lx**2 * m**2 * (-1 + M + M**2) * y**4 + (-1 + M**2 + k**2 * Lx**2 * m**2 * M * (2 + M)) * x * y**4)
                )
                
                g = lambda Lx, Ly, m, y, x: (np.exp(-((Ly * x) / (Lx * m * y))) * (Ly - y) * (Ly + y) * (Ly * x - Lx * m * y)) / (Lx**2 * m**2)
                    
                P = lambda Lx, Ly, m, y, k, x: (np.exp(1j * k * x - (Ly * x) / (Lx * m * y)) * (Ly - y) * (Ly + y) * (Ly * x - Lx * m * y)) / (Lx**2 * m**2)
                
                m_slope = 10
                self.__f = lambda x,y: eq_f(self.Lx, self.Ly, m_slope, y, pr.k0, x, pr.M0)
                self.__g = lambda x,y: g(self.Lx, self.Ly, m_slope, y, x) if y != 0 else 0
                self.__f_solution = lambda x,y: P(self.Lx, self.Ly, m_slope, y, pr.k0, x) if y != 0 else 0
            
            elif self.domain_shape == "array":
                raise ValueError("Not implemented the domain shape functions for array")
                
                
            elif self.domain_shape == "irregular":
                if self.operators_used == "basic":
                    raise ValueError("Basic operator. Irregular not implemented yet")
                
                elif self.operators_used == "complet":
                    if self.bc_side == "dirichlet":
                        
                        eq_f = lambda Lx, Ly, M, Omega, k, x, y: (
                            (1 / Lx**2) * np.exp(1j * k * x) * (
                                -2 * Lx * (Lx * (Ly - 3 * y) - 2 * Ly * x * (-1 + Omega))
                                - k**2 * (Ly - y) * (Lx * y + Ly * x * (-1 + Omega))**2
                                + 2j * k * M * (Ly - y) * (Lx * y + Ly * x * (-1 + Omega)) * (1j * k * Lx * y + Ly * (2 + 1j * k * x) * (-1 + Omega))
                                + (1 - M**2) * (Ly - y) * (
                                    k**2 * (Lx * y + Ly * x * (-1 + Omega))**2
                                    + 4j * k * Ly * (Lx * y + Ly * x * (-1 + Omega)) * (1 - Omega)
                                    - 2 * Ly**2 * (-1 + Omega)**2
                                )
                            )
                        )
                        
                        # d is the coefficient of the y^3 term
                        eq_g = g = lambda Lx, Ly, d, Omega, x, y: (
                            - (d * (Ly - y) * (Lx * y + Ly * x * (-1 + Omega))**2) / Lx**2
                        )
                        
                        d_y3 = 1
                        Omega = 0.5
                        
                        self.__g = lambda x, y: eq_g(self.Lx, self.Ly, d_y3, Omega, x, y)
                        
                        self.__f = lambda x, y: eq_f(self.Lx, self.Ly, pr.M0, Omega, pr.k0, x, y)
                        
                        self.__f_solution = lambda x, y: ( self.__g(x, y) * np.exp(1j * pr.k0 * x) )
                        
                    
                    elif self.bc_side == "neumann":
                        raise ValueError("Complete operator. Irregular not implemented yet for Neumann boundary condition")
                    
                    elif self.bc_side == "robin":
                        raise ValueError("Complete operator. Irregular not implemented yet for Robin boundary condition")
                    
                else:
                    raise ValueError("Invalid solution type")
                
            else:
                raise ValueError("Invalid domain shape")
            
        
    def build_matrixes(self):
        
        print("Building matrixes")
        
        start = time.time()
        
        if self.operators_used == "basic":
            A, b  = mtbasic.BC_onde_csr(self.__T, g = self.__g, metric = self.metric)
            A2,b2 = mtbasic.BC_up_down_csr(self.__T, self.__f, bc_side= self.bc_side ,metric = self.metric)
            A    += mtbasic.identite_csr(self.__T, scalar = pr.k0 ** 2, metric = self.metric) + mtbasic.laplacien_csr(self.__T, metric = self.metric)
            A    += mtbasic.BC_neumann_csr(self.__T, metric = self.metric) 
            b    += mtbasic.force_test(self.__T, f = self.__f, metric = self.metric)
            A += A2
            b += b2
        elif self.operators_used == "complet":
            A, b  = mtwhole.BC_onde_csr(self.__T, g = self.__g, metric = self.metric)
            A2,b2 = mtwhole.BC_up_down_csr(self.__T, self.__f, bc_side = self.bc_side, metric = self.metric)
            A    += mtwhole.identite_csr(self.__T, scalar = pr.k0 ** 2, metric = self.metric) + mtwhole.laplacien_csr(self.__T, metric = self.metric)
            A    += mtwhole.Dx2_csr(self.__T, scalar = -1 * pr.M0 ** 2, metric = self.metric) + mtwhole.Dx1_csr(self.__T, scalar = complex( 0, -2 * pr.M0 * pr.k0), metric = self.metric )
            A    += mtwhole.BC_neumann_csr(self.__T, metric = self.metric) 
            b    += mtwhole.force_test(self.__T, f = self.__f, metric = self.metric)
            A += A2
            b += b2
        else:
            raise ValueError("Invalid solution type")

        end = time.time()
        
        self.__time_matrix_build = end - start

        self.__A = A
        self.__b = b
        
        print("Finished Building matrixes")

    
    def solve_system (self):
        
        print("\nSolving system")
        
        start = time.time()
        
        ###### we are erasing the fixed nodes ######
        A = self.__A[self.__free_nodes, :]
        A = A[ : , self.__free_nodes]
        b = self.__b[self.__free_nodes]
        
        v = np.zeros((self.nodes_y * self.nodes_x), dtype=complex)
        v[self.__free_nodes] = spsolve(A, b)
        
        """ 
        Be careful with the reshape. I think there are sometimes
        that the reshape is not working as expected.
        I think it works row wise, so we have to transpose the matrix
        """
        self.__u = np.reshape(v, (self.nodes_x, self.nodes_y)).T
        
        end = time.time()
        self.__time_solve_system = end - start
        
        self.__real_error, self.__imaginary_error = anat.erreurs(self.u, f = self.__f_solution, metric = self.metric)
        
        self.__u_diff_real, self.__u_diff_imag = anat.erreur_sur_domaine(self.u, self.__real_error, self.__imaginary_error, f = self.__f_solution, metric = self.metric)
            
        if self.display:
        
            text = "Real errors: {:.2f}% | Imaginary errors: {:.2f}%\n".format(self.__real_error, self.__imaginary_error)
            text += "Time building matrix: {:.2f} s | Time solving system: {:.2f} s\n".format(self.__time_matrix_build, self.__time_solve_system)
            text += "Total time: {:.2f} s | h: {:.4f}".format(self.__time_matrix_build + self.__time_solve_system, self.metric.h)

            print("\n"+ text + "\n")
    
    
    def load_2_canva_numerical_solution(self):
        text = "Real errors: {:.2f}% | Imaginary errors: {:.2f}%\n".format(self.__real_error, self.__imaginary_error)
        text += "Time building matrix: {:.2f} s | Time solving system: {:.2f} s\n".format(self.__time_matrix_build, self.__time_solve_system)
        text += "Total time: {:.2f} s | h: {:.2f}".format(self.__time_matrix_build + self.__time_solve_system, self.metric.h)
        
        fig = plt.figure(figsize=(15, 9))
        ax1, fig = anat.plot_numerical_real(self.__u, fig, self.metric)
        ax2, fig = anat.plot_numerical_img(self.__u, fig, self.metric)
        # Ajouter une zone de texte en dessous de la figure
        fig.text(0.5, 0.05, text, ha='center', fontsize=10)
        
        if self.save:
            path = os.path.join("graphs", f"numerical_solution_{self.domain_shape}_{self.operators_used}_{self.bc_side}.png")
            fig.savefig(path, dpi = 400)
        
        
    def load_2_canva_analytical_solution(self):
        
        if self.final_setup:
            return
        
        fig = plt.figure(figsize=(15, 9))
        Z = np.zeros((self.nodes_y, self.nodes_x), dtype=complex)
        for i in range(self.nodes_y):
            for j in range(self.nodes_x):
                x = j * self.metric.hx
                y = self.metric.Ly - i * self.metric.hy
                
                Z[i,j]= self.__f_solution(x, y)
                
        # Obtenez la partie réelle et la partie imaginaire de Z
        Z_real = np.real(Z)
        Z_imag = np.imag(Z)
        
        X, Y = np.meshgrid(np.linspace(0, self.metric.Lx, self.nodes_x), np.linspace(0, self.metric.Ly, self.nodes_y))
        
        # Graphique de la partie réelle
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(X, Y, Z_real, cmap='coolwarm', edgecolor='none')
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z_real")
        fig.colorbar(surf1, ax=ax1)

        # Graphique de la partie imaginaire
        ax2 = fig.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(X, Y, Z_imag, cmap='coolwarm', edgecolor='none')
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z_imag")
        fig.colorbar(surf2, ax=ax2)
        # Ajouter des titres aux graphiques de la deuxième figure
        ax1.set_title("partie réelle analityque")
        ax2.set_title("partie imaginaire analityque")
        
        if self.save:
            path = os.path.join("graphs", f"analytical_solution_{self.domain_shape}_{self.operators_used}_{self.bc_side}.png")
            fig.savefig(path, dpi = 400)
                

    def load_2_canva_diff_solution (self):
        fig = plt.figure(figsize=(15, 9))

        # Créer les grilles X et Y de manière appropriée
        X, Y = np.meshgrid(np.linspace(0, self.metric.Lx, self.nodes_x), np.linspace(0, self.metric.Ly, self.nodes_y))

        # Afficher la solution u en 3D (partie réelle)
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(X, Y, self.__u_diff_real, cmap='viridis', edgecolor='none')
        fig.colorbar(surf1, ax=ax1)
        
        #ax1.set_box_aspect([np.ptp(X), np.ptp(Y), min(np.ptp(X), np.ptp(Y))])
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('u_real')
        ax1.set_title('Partie réelle de la difference u')

        # Afficher la solution u en 3D (partie imaginaire)
        ax2 = fig.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(X, Y, self.__u_diff_imag, cmap='viridis', edgecolor='none')
        fig.colorbar(surf2, ax=ax2)
        
        #ax2.set_box_aspect([np.ptp(X), np.ptp(Y), min(np.ptp(X), np.ptp(Y))])
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('u_imag')
        ax2.set_title('Partie imaginaire de la difference u')        
        
        if self.save:
            path = os.path.join("graphs", f"Difference_solution_{self.domain_shape}_{self.operators_used}_{self.bc_side}.png")
            fig.savefig(path, dpi = 400)


    def load_combined_canvas(self):
        text = "Real errors: {:.2f}% | Imaginary errors: {:.2f}%\n".format(self.__real_error, self.__imaginary_error)
        text += "Time building matrix: {:.2f} s | Time solving system: {:.2f} s\n".format(self.__time_matrix_build, self.__time_solve_system)
        text += "Total time: {:.2f} s | h: {:.2f}".format(self.__time_matrix_build + self.__time_solve_system, self.metric.h)
        
        fig = plt.figure(figsize=(19, 9.5))        
        
        # Analytical solution plots
        Z = np.zeros((self.nodes_y, self.nodes_x), dtype=complex)
        for i in range(self.nodes_y):
            for j in range(self.nodes_x):
                x = j * self.metric.hx
                y = self.metric.Ly - i * self.metric.hy
                Z[i, j] = self.__f_solution(x, y)
        Z_real = np.real(Z)
        Z_imag = np.imag(Z)
        X, Y = np.meshgrid(np.linspace(0, self.metric.Lx, self.nodes_x), np.linspace(0, self.metric.Ly, self.nodes_y))
        
        
        # Afficher la solution u en 3D (partie réelle)
        ax1 = fig.add_subplot(321, projection='3d')
        surf1 = ax1.plot_surface(X, Y, np.real(self.__u), cmap='viridis', edgecolor='none')
        fig.colorbar(surf1, ax=ax1)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('u_real')
        ax1.set_title('Numerical Real')
        
        # Afficher la solution u en 3D (partie imaginaire)
        ax2 = fig.add_subplot(322, projection='3d')
        surf2 = ax2.plot_surface(X, Y, np.imag(self.__u), cmap='viridis', edgecolor='none')
        fig.colorbar(surf2, ax=ax2)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('u_imag')
        ax2.set_title('Numerical Imaginary')
        

        ax3 = fig.add_subplot(323, projection='3d')
        surf3 = ax3.plot_surface(X, Y, Z_real, cmap='coolwarm', edgecolor='none')
        fig.colorbar(surf3, ax=ax3)
        ax3.set_title('Analytical Real')

        ax4 = fig.add_subplot(324, projection='3d')
        surf4 = ax4.plot_surface(X, Y, Z_imag, cmap='coolwarm', edgecolor='none')
        fig.colorbar(surf4, ax=ax4)
        ax4.set_title('Analytical Imaginary')
        
        # Difference solution plots
        ax5 = fig.add_subplot(325, projection='3d')
        surf5 = ax5.plot_surface(X, Y, self.__u_diff_real, cmap='plasma', edgecolor='none')
        fig.colorbar(surf5, ax=ax5)
        ax5.set_title('Difference Real')

        ax6 = fig.add_subplot(326, projection='3d')
        surf6 = ax6.plot_surface(X, Y, self.__u_diff_imag, cmap='plasma', edgecolor='none')
        fig.colorbar(surf6, ax=ax6)
        ax6.set_title('Difference Imaginary')
        
        # Add the text below the figure
        fig.text(0.5, 0.01, text, ha='center', fontsize=12)
        
        if self.save:
            path = os.path.join("graphs", f"combined_solution_{self.domain_shape}_{self.operators_used}_{self.bc_side}.png")
            fig.savefig(path, dpi=400)
        
        plt.show()




#### another function #######

def plot_error_vs_h(real_error, img_error, h):

    # Create a figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    
    dh = list(map(lambda x: 1 / x, h))

    # Plot h vs real_error
    ax1.plot(dh, real_error, marker='o', linestyle='-')
    ax1.set_xlabel('1/h')
    ax1.set_ylabel('Real Error')
    ax1.set_title('1/h vs Real Error')
    ax1.set_xscale('log')  # Set logarithmic scale for h
    ax1.set_yscale('log')  # Set logarithmic scale for h

    # Plot h vs img_error
    ax2.plot(dh, img_error, marker='o', linestyle='-')
    ax2.set_xlabel('1/h')
    ax2.set_ylabel('Image Error')
    ax2.set_title('1/h vs Image Error')
    ax2.set_xscale('log')  # Set logarithmic scale for h
    ax2.set_yscale('log')  # Set logarithmic scale for h

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()


            


if __name__ == "__main__":
    
    init_params = {
        "Lx": 2,
        "Ly": 2,
        "display": True,
        "bc_side": 1,               # 0: "dirichlet" ou 1: "neumann"
        "operators_used": 1,        # 0: "basic" ou 1: "complet"
        "domain_shape": 0,          # we can choose: 0: rect - 1: img - 2: array
        "multiplier_4_increasing_resolution": 10,
        "display" : True,
        "save" : False,
    }

    system = SystemBuilder(**init_params)
    system.build_matrixes()
    system.solve_system()
    
    
    system.load_2_canva_numerical_solution()
    system.load_2_canva_analytical_solution()
    
    plt.show()
        
        

    