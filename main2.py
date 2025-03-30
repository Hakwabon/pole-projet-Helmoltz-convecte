from system_builder import SystemBuilder
from matplotlib import pyplot as plt
import numpy as np
import parameters as pr
import affichage as aff

init_params = {
    "Lx": 50,
    "Ly": 50,
    "display": True,
    "bc_side": 0,               # 0: "dirichlet" ou 1: "robin" ou 2: "neumann"
    "operators_used": 0,        # 0: "basic" ou 1: "complet"
    "domain_shape": 1,          # we can choose: 0: rect - 1: img - 2: array
    "final_setup" : False,       # will use f = 0 and g = y ( L - y )
    "multiplier_4_increasing_resolution": 6,
    "display" : True,
    "save" : False,
    "image_path" : "Irregulier.png",
}
# init_params = aff.traitement()
# print(init_params)

system = SystemBuilder(**init_params)
system.build_matrixes()
system.solve_system()

system.load_2_canva_diff_solution()
system.load_2_canva_numerical_solution()
system.load_2_canva_analytical_solution()

# system.load_combined_canvas()

plt.show()