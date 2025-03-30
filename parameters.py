
#k = w/v w = [40, 120]Hz v=M0*c c = [300, 350] T = [0,40] degres Celscius M0 = [0.1,0.8]


M0 = 0.1
# k = 60/(M0*320)
k = 0.2
k0 = k
Z0 = 1
Z = complex(1, 0)

eta = Z0 / Z
#eta = 0

alpha = k0 * Z0 / Z
alphaR = alpha.real 
alphaI = alpha.imag
hx = 1  # Pas de discrétisation en x
hy = 1  # Pas de discrétisation en y
Lx = 100
Ly = 100

dx = 0.001
dy = 0.001


