import numpy as np
from parameters import *
import fonction_analytique as anat
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import inv
from typing import List, Union, Callable, Optional
from os import path
from typing import Optional, Callable


""" 

This script is for using the new implementation of the matrix operators
but with dirichelt border condition and wihtout the complex terms in the main equation

"""


class CSRBuilder:
    def __init__(self, T : np.ndarray = None, nodes_y : int = None, nodes_x : int = None):
        
        if T is not None:
            nodes_y, nodes_x = T.shape
        else:
            assert nodes_y is None or nodes_x is None
        
        self.nodes_y = nodes_y
        self.nodes_x = nodes_x
        self.data = []
        self.row_indexes = []
        self.col_indexes = []
        self.grid_size = nodes_y * nodes_x
        
    
    def get_node(self, row : int, col : int) -> int:
        return int(row + col * self.nodes_y)
        
    
    # this are the nodes of the big matrix that we want to build
    def add_value(self, row_big_matrix: int, col_big_matrix: int, value: complex):
        self.data.append(value)
        self.row_indexes.append(row_big_matrix)
        self.col_indexes.append(col_big_matrix)

    """ 
    The add value xxxx recieve the row and col of the node and the value to add
    Remember that the matrixes are built as column wise and downwards, e.i., 
    node + 1 = node down
    node - 1 = node up
    node + nodes_y = node right
    node - nodes_y = node left
    """
    
    def add_value_center(self, row: int, col: int, value: complex):
        node = self.get_node(row, col)
        node_value = self.get_node(row, col) 
        self.add_value(node, node_value, value)

    def add_value_right(self, row: int, col: int, value: complex):
        node = self.get_node(row, col)
        node_value = self.get_node(row, col + 1) 
        self.add_value(node, node_value, value)

    def add_value_left(self, row: int, col: int, value: complex):
        node = self.get_node(row, col)
        node_value = self.get_node(row, col - 1) 
        self.add_value(node, node_value, value)

    def add_value_up(self, row: int, col: int, value: complex):
        node = self.get_node(row, col)
        node_value = self.get_node(row - 1, col) 
        self.add_value(node, node_value, value)

    def add_value_down(self, row: int, col: int, value: complex):
        node = self.get_node(row, col)
        node_value = self.get_node(row + 1, col) 
        self.add_value(node, node_value, value)

    def build(self):
        data = np.array(self.data, dtype=complex)
        row_indexes = np.array(self.row_indexes)
        col_indexes = np.array(self.col_indexes)

        return csr_matrix((data, (row_indexes, col_indexes)), shape=(self.grid_size, self.grid_size))


def laplacien_csr(T : np.ndarray, metric : anat.Metric = None) -> csr_matrix:
    assert T.ndim == 2  #matrix must be 2D
    csr_builder = CSRBuilder(T)
    
    print("Building laplacien csr")
    
    hx = metric.hx
    hy = metric.hy
        
    for i in range(csr_builder.nodes_y):
        for j in range(csr_builder.nodes_x):
            if T[i, j] == 2:
        
                csr_builder.add_value_center(i, j, -2.0 / (hx * hx) - 2.0 / (hy * hy))
                
                if i > 0:
                    csr_builder.add_value_up(i, j, 1.0 / (hx * hx))
                if i < csr_builder.nodes_y - 1:
                    csr_builder.add_value_down(i, j, 1.0 / (hx * hx))
                if j > 0:
                    csr_builder.add_value_left(i, j, 1.0 / (hy * hy))
                if j < csr_builder.nodes_x - 1:
                    csr_builder.add_value_right(i, j, 1.0 / (hy * hy))

    return csr_builder.build()


def identite_csr(T : np.ndarray, scalar: complex = None, metric : anat.Metric = None ) -> csr_matrix:
    assert T.ndim == 2  #matrix must be 2D
    
    print("Building identite csr")
    
    if scalar is None:
        scalar = 1

    csr_builder = CSRBuilder(T)
    
    for i in range(csr_builder.nodes_y):
        for j in range(csr_builder.nodes_x):
            if T[i, j] == 2:
                csr_builder.add_value_center(i, j, scalar)
    return csr_builder.build()


def Dx2_csr (T: np.ndarray, scalar = 1, metric : anat.Metric = None) -> csr_matrix:
    assert T.ndim == 2  #matrix must be 2D
    
    if metric is None:
        raise ValueError("Dx2. Metric is None")
    
    hx = metric.hx
    hy = metric.hy
    
    print("Building Dx2 csr")
    
    csr_builder = CSRBuilder(T)
    for i in range(csr_builder.nodes_y):
        for j in range(csr_builder.nodes_x):
            if T[i, j] == 2:
                csr_builder.add_value_center(i, j, -2.0 * scalar / hx ** 2)
                
                if j > 0:
                    csr_builder.add_value_left(i, j, scalar / hx ** 2)
                if j < csr_builder.nodes_x - 1:
                    csr_builder.add_value_right(i, j, scalar / hx ** 2)
    return csr_builder.build()      


def Dx1_csr (T: np.ndarray, scalar = 1, metric : anat.Metric = None) -> csr_matrix:
    assert T.ndim == 2  #matrix must be 2D
    
    if metric is None:
        raise ValueError("Dx1. Metric is None")
    
    hx = metric.hx
    hy = metric.hy
    h = metric.h
    
    print("Building Dx1 csr")
    
    csr_builder = CSRBuilder(T)
    for i in range(csr_builder.nodes_y):
        for j in range(csr_builder.nodes_x):
            if T[i, j] == 2:
                csr_builder.add_value_left(i, j, -1 * scalar / (2 * hx))
                csr_builder.add_value_right(i, j, scalar / (2 * hx))   
                if j == 0:
                    raise ValueError("Dx1. The left boundary is not part of the domain")
                if j == csr_builder.nodes_x - 1:
                    raise ValueError("Dx1. The right boundary is not part of the domain")
                              
    return csr_builder.build()  


def force_test(T: np.ndarray, f: Optional[Callable[[int, int], float]] = None, metric : anat.Metric = None) -> np.ndarray:
    assert T.ndim == 2  #matrix must be 2D
    
    print("Building force test csr")
    
    nodes_y, nodes_x = T.shape
    b = np.zeros((nodes_y*nodes_x),dtype=complex)
    
    hx = metric.hx
    hy = metric.hy
    h = metric.h
    
    
    if f is None:
        f = lambda x, y: -2 * np.exp(complex(0, k0 * x))
    
    for i in range(nodes_y):
        for j in range(nodes_x):
            k = i + j * nodes_y
            x = j * hx
            y = metric.Ly - i * hy
            
            if int(T[i,j]) == 2 or int(T[i,j]) in [7, 7.0]:
                b[k] += f(x, y)    
    return b


def BC_neumann_csr(T, metric : anat.Metric = None) -> csr_matrix:
    
    """ 
    The number 7 is the number of the boundary condition in the matrix T
    and we add 1 ... 8 to indicate direction of the normal vector to the boundary
    
    1  2  3
    4  X  5
    6  7  8
    
    To apply this solution, we consider hx == hy
    """
    
    print("Building BC neumann csr")
    
    hx = metric.hx
    hy = metric.hy
    h = metric.h
    
    csr_builder = CSRBuilder(T)
    
    for row in range(csr_builder.nodes_y):
        for col in range(csr_builder.nodes_x):
            
            
            #if T[row, col] == 74:
            if int(T[row, col]) in [7, 7.0, 70]:
                
                csr_builder.add_value_center(row, col, k0**2 * (1 + 2 * M0) + 2 * (-2 + M0**2) / h**2 + 2j *k0 * (-1 + M0**2) / h )
                csr_builder.add_value_right(row, col, 2*(1 - M0**2) / h ** 2)
                csr_builder.add_value_up(row, col, 1.0 / h ** 2)
                csr_builder.add_value_down(row, col, 1.0 / h ** 2) 
                                        
    return csr_builder.build()


def BC_onde_csr(T, g : Optional[Callable[[int, int], float]] = None, metric : anat.Metric = None) -> Union[csr_matrix, csr_matrix]:
    
    """
    For having a sense of direction
    in here we are also going to use
    the following system of reference
    
    1  2  3
    4  X  5
    6  7  8
    """
    
    print("Building BC onde csr")
    
    hx = metric.hx
    hy = metric.hy
    h = metric.h
      
    nodes_y, nodes_x = T.shape
    b = np.zeros((nodes_y * nodes_x), dtype=complex)
    csr_builder = CSRBuilder(T)
    
    if g is None:
        g = lambda x, y: 1


    for row in range(nodes_y):
        for col in range(nodes_x):
            k = row + col * nodes_y  # Indice de la cellule (i, j)
            x = col * hx
            y = metric.Ly - row * hy

            if T[row, col] in list(range(81, 89)) or T[row, col] in [8, 8.0]:
                b[k] += np.exp(complex(0, k0 * x)) * g(x, y)
                csr_builder.add_value_center(row, col, 1)
        
    matrix = csr_builder.build()

    return matrix, b


def BC_up_down_csr(T, f = None, bc_side : str = None, metric : anat.Metric = None):

    print("Building BC up and down sides csr")
    
    h = metric.h
    hx = metric.hx
    hy = metric.hy
    
    nodes_y, nodes_x = T.shape
    b = np.zeros((nodes_y * nodes_x), dtype=complex)
    csr_builder = CSRBuilder(T)
    
    #eta = Z0 / Z
    #eta = 0
    
    if bc_side == "robin":
        eta = Z0 / Z
    elif bc_side == "neumann":
        eta = 0

    
    for row in range(csr_builder.nodes_y):
        for col in range(csr_builder.nodes_x):
            index = row + col * nodes_y
            x = col * hx
            y = metric.Ly - row * hy
            
            """ Here im going to have the border conditions for dirichelt and neumann """
            
            if bc_side == "dirichlet":
                if T[row, col] in list(range(71, 79)) + [91, 96]:                
                    b[index] += 0
                    csr_builder.add_value_center(row, col, 1)
            
            elif bc_side == "robin" or bc_side == "neumann":
                if T[row, col] in list(range(71, 79)) + [91, 96]:  
                    b[index] += f(x, y)
                    
                
                if T[row, col] == 91:
                    csr_builder.add_value_center(row, col,  (k0**2 * (1 + 2 * M0) - (4j * M0**2 * eta) / (h**3 * k0) + (2j * k0 * (-1 + M0**2 - eta - 2 * M0 * eta)) / h + (-4 + M0**2 * (2 + 4 * eta)) / h**2) )
                    csr_builder.add_value_down(row, col, 2/h**2)
                    csr_builder.add_value_right(row, col, (-2 * h * k0 * (-1 + M0**2) + 4j * M0**2 * eta) / (h**3 * k0) )
                    
                if T[row, col] == 96:
                    csr_builder.add_value_center(row, col, (k0**2 * (1 + 2 * M0)-(4j * M0**2 * eta) / (h**3 * k0)+(2j * k0 * (-1 + M0**2 - eta - 2 * M0 * eta)) / h+(-4 + M0**2 * (2 + 4 * eta)) / h**2))
                    csr_builder.add_value_up(row, col, 2/h**2)
                    csr_builder.add_value_right(row, col, (-2 * h * k0 * (-1 + M0**2) + 4j * M0**2 * eta) / (h**3 * k0))
                
                    
                if T[row, col] == 71: #Upper left
                    if row == metric.nodes_y - 1 or col == metric.nodes_x - 1:
                        raise ValueError("BC up down csr. The upper left 71 boundary is not part of the domain")    
                    
                    csr_builder.add_value_center(row, col, -k0*complex(-4+h**2*k0**2 + 2*M0**2 + 8*M0*eta, -4*h*k0*eta) /h /complex(h*k0*(-1+2*M0*eta), 2*M0**2*eta) )
                    csr_builder.add_value_down(row, col, 2/h**2)
                    csr_builder.add_value_right(row, col, complex(2*h*k0*(-1+M0**2+2*M0*eta), -4*M0**2*eta ) / h**2 /complex(h*k0*(-1+2*M0*eta), 2*M0**2*eta )  )
                    
                elif T[row, col] == 72: # up
                    if row == metric.nodes_y - 1 or col == metric.nodes_x - 1 or col == 0: 
                        raise ValueError("BC up down csr. The upper 72 boundary is not part of the domain")
                    csr_builder.add_value_center(row, col, complex( (2*M0**2-4)/h**2 + k0**2, -4*eta*M0**2/(k0*h**3) - 2*k0*eta/h))
                    csr_builder.add_value_down(row, col, 2 / h**2)
                    csr_builder.add_value_right(row, col, complex( (1 - M0**2 - 2*eta*M0)/h**2, -k0*M0/h + 2*eta*M0**2/(k0 * h**3)))
                    csr_builder.add_value_left(row, col, complex( (1 - M0**2 + 2*eta*M0)/h**2, k0*M0/h + 2*eta*M0**2/(k0 * h**3)))
                    
                elif T[row, col] ==73: # Upper right
                    if row == metric.nodes_y - 1 or col == 0:
                        raise ValueError("BC up down csr. The upper right 73 boundary is not part of the domain")
                    
                    csr_builder.add_value_center(row, col, k0*complex(-4+h**2*k0**2 + 2*M0**2 - 8*M0*eta , -4*h*k0*eta ) / h / complex( h*(k0+2*k0*M0*eta) , -2*M0**2*eta ))
                    csr_builder.add_value_down(row, col, 2/h**2)
                    csr_builder.add_value_left(row, col, complex(-2*h*k0*(-1+M0**2-2*M0*eta), 4*M0**2*eta) / h**2 / complex( h*(k0 + 2*k0*M0*eta), -2*M0**2*eta ) )
                    
                elif T[row, col] == 74: #left
                    if col == metric.nodes_x - 1:
                        raise ValueError("BC up down csr. The left 74 boundary is not part of the domain")
                    
                    csr_builder.add_value_center(row, col, complex(h**3 * k0**3 + 2*h*k0*(M0**2 - 2 - 4*M0*eta), 2*h**2*k0**2*eta-4*M0**2*eta) / h**2 / complex(h*(k0+2*k0*M0*eta), 2*M0**2*eta))
                    csr_builder.add_value_up(row, col, 1 / h**2)
                    csr_builder.add_value_down(row, col, 1 / h**2)
                    csr_builder.add_value_right(row, col, -2*k0/h*(M0**2-1-2*M0*eta) / complex(h*(k0+2*k0*M0*eta), 2*M0**2*eta))
                    
                elif T[row, col] == 75: #right
                    if col == 0:
                        raise ValueError("BC up down csr. The right 75 boundary is not part of the domain")
                    
                    csr_builder.add_value_center(row, col, complex(h**3*k0**3 + 2*h*k0*(-2 + M0**2 - 4*M0*eta, -2*h**2*k0**2*eta + 4*M0**2*eta)) / h**2 / complex(h* (k0 + 2*k0*M0*eta) , -2*M0**2*eta))
                    csr_builder.add_value_down(row, col, 1 / h**2)
                    csr_builder.add_value_up(row, col, 1/h**2)
                    csr_builder.add_value_left(row, col, -2*k0*(-1 + M0**2 - 2*M0*eta)/h/complex(h*(k0 + 2*k0*M0*eta) , -2*M0**2*eta ) )
                    
                elif T[row, col] == 76: # Lower left
                    if row == 0 or col == metric.nodes_x - 1:
                        raise ValueError("BC up down csr. The lower left 76 boundary is not part of the domain")
                    
                    csr_builder.add_value_center(row, col, -k0 * complex(-4 + h**2*k0**2 + 2*M0**2 + 8*M0*eta, -4*h*k0*eta) / h / complex(h*k0*(-1 + 2*M0*eta), 2*M0**2*eta) )
                    csr_builder.add_value_up(row, col, 2 / h**2)
                    csr_builder.add_value_right(row, col, complex(2*h*k0*(-1 + M0**2 + 2*M0*eta), -4*M0**2*eta) / h**2 / complex(h*k0*(-1+2*M0*eta), 2*M0**2*eta))
                    
                elif T[row, col] == 77: # down
                    if row == 0 or col == metric.nodes_x - 1 or col == 0:
                        raise ValueError("BC up down csr. The down 77 boundary is not part of the domain")
                    
                    csr_builder.add_value_center(row, col, complex( (2*M0**2-4)/h**2 + k0**2, -4*eta*M0**2/(k0*h**3) - 2*k0*eta/h))
                    csr_builder.add_value_up(row, col, 2 / h**2)
                    csr_builder.add_value_right(row, col, complex( (1 - M0**2 - 2*eta*M0)/h**2, - k0*M0/h + 2*eta*M0**2/(k0 * h**3)))
                    csr_builder.add_value_left(row, col, complex( (1 - M0**2 + 2*eta*M0)/h**2, k0*M0/h + 2*eta*M0**2/(k0 * h**3)))
                
                elif T[row, col] == 78: # Lower right
                    if row == 0 or col == 0:
                        raise ValueError("BC up down csr. The lower right 78 boundary is not part of the domain")
                    
                    csr_builder.add_value_center(row, col, k0*complex(-4 + h**2*k0**2 + 2*M0**2 - 8*M0*eta, -4*h*k0*eta ) / h / complex(h*(k0 + 2*k0*M0*eta), -2*M0**2*eta) )
                    csr_builder.add_value_up(row, col, 2 / h**2)
                    csr_builder.add_value_left(row, col, complex(-2*h*k0*(-1 + M0**2 + 2*M0*eta), 4*M0**2*eta) / h**2 / complex( h*(k0 + 2*k0*M0*eta), -2*M0**2*eta ))
                   
                                                        
    matrix = csr_builder.build()        
    return matrix, b

