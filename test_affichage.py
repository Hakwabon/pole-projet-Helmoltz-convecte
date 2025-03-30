import tkinter as tk
from tkinter import ttk

class SimulationApp:
    def __init__(self):
        self.values = None

    def affichage(self):
        root = tk.Tk()
        root.title("Simulation Parameters")

        def start_simulation():
            solution = solution_var.get()
            print("Solution:", solution)
            if solution == "Complet":
                Complexe_window()
            elif solution == "Basic":
                Basic_window()

        def Basic_window():
            new_window = tk.Toplevel(root)
            new_window.title("Additional Parameters")

            Lx_label = ttk.Label(new_window, text="Lx:")
            Lx_label.grid(row=0, column=0)
            Lx_scale = tk.Scale(new_window, from_=10, to=200, orient=tk.HORIZONTAL)
            Lx_scale.set(50)
            Lx_scale.grid(row=0, column=1, sticky="w")

            Ly_label = ttk.Label(new_window, text="Ly:")
            Ly_label.grid(row=1, column=0)
            Ly_scale = tk.Scale(new_window, from_=10, to=200, orient=tk.VERTICAL)
            Ly_scale.set(100)
            Ly_scale.grid(row=1, column=1, sticky="w")

            force_var = tk.StringVar()
            force_label = ttk.Label(new_window, text="Force (second membre):")
            force_label.grid(row=2, column=0, sticky="w")
            force_combo = ttk.Combobox(new_window, textvariable=force_var, values=["Test", "Irregular", "0"])
            force_combo.set("Irregular")
            force_combo.grid(row=2, column=1, sticky="w")

            m_label = ttk.Label(new_window, text="Pourcentage de pente pour la force irrégulière (fois 5):")
            m_label.grid(row=3, column=0)
            m_scale = tk.Scale(new_window, from_=1, to=10, orient=tk.HORIZONTAL)
            m_scale.set(1)
            m_scale.grid(row=3, column=1, sticky="w")

            fig_anat_var = tk.StringVar()
            fig_anat_label = ttk.Label(new_window, text="Figure (comparaison analytique ou solution réacteur):")
            fig_anat_label.grid(row=5, column=0, sticky="w")
            fig_anat_combo = ttk.Combobox(new_window, textvariable=fig_anat_var, values=["Comparaison", "Reacteur"])
            fig_anat_combo.set("Comparaison")
            fig_anat_combo.grid(row=5, column=1, sticky="w")

            multiplier_4_increasing_resolution_label = ttk.Label(new_window, text="Multiplier for Increasing Resolution:")
            multiplier_4_increasing_resolution_label.grid(row=6, column=0, sticky="w")
            multiplier_4_increasing_resolution_scale = tk.Scale(new_window, from_=1, to=10, orient=tk.HORIZONTAL)
            multiplier_4_increasing_resolution_scale.set(4)
            multiplier_4_increasing_resolution_scale.grid(row=6, column=1, sticky="w")

            def apply_and_close():
                Lx = Lx_scale.get()
                Ly = Ly_scale.get()
                m = m_scale.get() * 5
                force = force_var.get()
                fig_anat = fig_anat_var.get()
                multiplier_4_increasing_resolution = multiplier_4_increasing_resolution_scale.get()
                print("Lx:", Lx)
                print("Ly:", Ly)
                print("Force:", force)
                print("Pourcentage de pente:", m)
                print("Fig_anat:", fig_anat)
                print("Multiplier for increasing resolution:", multiplier_4_increasing_resolution)
                self.values = {
                    "Lx": Lx,
                    "Ly": Ly,
                    "bc_side": None,
                    "fig_anat": fig_anat,
                    "multiplier": multiplier_4_increasing_resolution,
                    "solution": "Basic",
                    "pente": m,
                    "Force": force,
                    "forme": None
                }
                new_window.destroy()
                root.quit()

            apply_button = ttk.Button(new_window, text="Apply", command=apply_and_close)
            apply_button.grid(row=7, columnspan=2, pady=10)

        def Complexe_window():
            new_window = tk.Toplevel(root)
            new_window.title("Additional Parameters")

            Lx_label = ttk.Label(new_window, text="Lx:")
            Lx_label.grid(row=0, column=0)
            Lx_scale = tk.Scale(new_window, from_=10, to=200, orient=tk.HORIZONTAL)
            Lx_scale.set(50)
            Lx_scale.grid(row=0, column=1, sticky="w")

            Ly_label = ttk.Label(new_window, text="Ly:")
            Ly_label.grid(row=1, column=0)
            Ly_scale = tk.Scale(new_window, from_=10, to=200, orient=tk.VERTICAL)
            Ly_scale.set(100)
            Ly_scale.grid(row=1, column=1, sticky="w")

            bc_side_var = tk.StringVar()
            bc_side_label = ttk.Label(new_window, text="Condition aux bords (0,dérivé normal,condition complète):")
            bc_side_label.grid(row=2, column=0, sticky="w")
            bc_side_combo = ttk.Combobox(new_window, textvariable=bc_side_var, values=["Dirichlet", "Neumann", "Robin"])
            bc_side_combo.set("Dirichlet")
            bc_side_combo.grid(row=2, column=1, sticky="w")

            shape_var = tk.StringVar()
            shape_label = ttk.Label(new_window, text="Forme du domaine :")
            shape_label.grid(row=3, column=0, sticky="w")
            shape_combo = ttk.Combobox(new_window, textvariable=shape_var, values=["carré", "irrégulier"])
            shape_combo.set("carré")
            shape_combo.grid(row=3, column=1, sticky="w")

            fig_anat_var = tk.StringVar()
            fig_anat_label = ttk.Label(new_window, text="Figure (comparaison analytique ou solution):")
            fig_anat_label.grid(row=5, column=0, sticky="w")
            fig_anat_combo = ttk.Combobox(new_window, textvariable=fig_anat_var, values=["Comparaison", "Reacteur"])
            fig_anat_combo.set("Comparaison")
            fig_anat_combo.grid(row=5, column=1, sticky="w")

            multiplier_4_increasing_resolution_label = ttk.Label(new_window, text="Multiplier for Increasing Resolution:")
            multiplier_4_increasing_resolution_label.grid(row=6, column=0, sticky="w")
            multiplier_4_increasing_resolution_scale = tk.Scale(new_window, from_=1, to=10, orient=tk.HORIZONTAL)
            multiplier_4_increasing_resolution_scale.set(4)
            multiplier_4_increasing_resolution_scale.grid(row=6, column=1, sticky="w")

            def apply_and_close():
                Lx = Lx_scale.get()
                Ly = Ly_scale.get()
                bc_side = bc_side_var.get()
                shape = shape_var.get()
                fig_anat = fig_anat_var.get()
                multiplier_4_increasing_resolution = multiplier_4_increasing_resolution_scale.get()
                print("Lx:", Lx)
                print("Ly:", Ly)
                print("BC side:", bc_side)
                print("Forme:", shape)
                print("Fig_anat:", fig_anat)
                print("Multiplier for increasing resolution:", multiplier_4_increasing_resolution)
                self.values = {
                    "Lx": Lx,
                    "Ly": Ly,
                    "bc_side": bc_side,
                    "fig_anat": fig_anat,
                    "multiplier": multiplier_4_increasing_resolution,
                    "solution": "Complet",
                    "pente": False,
                    "Force": False,
                    "forme": shape
                }
                new_window.destroy()
                root.quit()

            apply_button = ttk.Button(new_window, text="Apply", command=apply_and_close)
            apply_button.grid(row=7, columnspan=2, pady=10)

        solution_var = tk.StringVar()
        solution_label = ttk.Label(root, text="Solution:")
        solution_label.grid(row=0, column=0, sticky="w")
        solution_combo = ttk.Combobox(root, textvariable=solution_var, values=["Basic", "Complet"])
        solution_combo.set("Basic")
        solution_combo.grid(row=0, column=1, sticky="w")

        start_button = ttk.Button(root, text="Start", command=start_simulation)
        start_button.grid(row=1, columnspan=2, pady=10)

        root.mainloop()
        return self.values

def traitement():
    app = SimulationApp()
    values = app.affichage()

    if values is None:
        raise ValueError("No values returned from the simulation setup")

    param = {}

    param["Lx"], param["Ly"], param["display"], param["multiplier_4_increasing_resolution"], param["save"] = values["Lx"], values["Ly"], True, values["multiplier"], False

    if values["solution"] == "Basic":
        param["operators_used"] = 0
        if values["fig_anat"] == "Comparaison":
            param["bc_side"], param["final_setup"] = 0, False
            if values["Force"] == "Test":
                param["domain_shape"] = 0
                param["image_path"] = None
            elif values["Force"] == "Irregular":
                param["domain_shape"] = 1
                param["image_path"] = "Irregulier.png"
            elif values["Force"] == 0:
                param["final_setup"] = True
                param["domain_shape"] = 0
                param["image_path"] = None
        elif values["fig_anat"] == "Reacteur":
            param["bc_side"], param["domain_shape"], param["final_setup"], param["image_path"] = 1, 1, True, "réacteur2.png"

    elif values["solution"] == "Complet":
        param["operators_used"] = 1
        if values["fig_anat"] == "Comparaison":
            param["final_setup"] = False
            if values["bc_side"] == "Dirichlet":
                param["bc_side"] = 0
            elif values["bc_side"] == "Neumann":
                param["bc_side"] = 2
            elif values["bc_side"] == "Robin":
                param["bc_side"] = 1
            if values["forme"] == "carré":
                param["domain_shape"], param["image_path"] = 0, None
            elif values["forme"] == "irrégulier":
                param["domain_shape"], param["image_path"] = 1, "Irregulier.png"
        elif values["fig_anat"] == "Reacteur":
            param["final_setup"], param["image_path"] = True, "réacteur2.png"
            if values["bc_side"] == "Robin":
                param["bc_side"] = 1
            elif values["bc_side"] == "Neumann":
                param["bc_side"] = 2
            else:
                param["bc_side"] = 1

    return param

print(traitement())
