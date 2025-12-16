"""
Lower Limb Torque and Force Capacity Visualization GUI
Frailty - Stair Climbing Analysis


Data structure:
(root folder)\
    gui_stair_climbing.py
    data\
        momentArm_7x92x120.mat                          (moment arms for 7DOF x 92 muscles x 120 timescale)
        force_92x120.mat                                (max muscle forces for 92 muscles x 120 timescale)
        inverse_dynamics_scaled.sto                     (inverse dynamics for torque and force demands of the task)
        AB09_stair_s20dg_03.txt                         (joint angles, 7DOF x 120 timescale)
        segment_lengths.txt                             (3DOF x 1 length per relevant segment--hip, knee, ankle only)
        3DGaitModel2392_StaticOptimization_force.sto    (for getting muscle names)
        jointTorque_7x120.mat                           (for verification)


# Generative AI was used to provide a workflow framework and assist with coding.
# Tool used: Gemini 3 Pro
# Initial prompt: "[We are] trying to model the human lower limb to study the diminishing feasible output forces and parameters as a person grows elderly, framing the problem in the context of what [we] learned in class. . . about torque, force, and muscle activation spaces.
# Also attached is a pdf compiling [our] research notes, resources, and tasks.
# Describe thoroughly, the GUI producer's role, and outline their workflow steps:
# "The other to write a script to visualize the output-figures to a video (for example, make a GUI, Users can choose muscles they interested in, then by dragging the progress bar, users can choose each leg posture, and corresponding feasible force set will appear next to the posture)."
# Make sure to also describe the required input parameters that the script will accept and read, before producing the user-interactive GUI.
# Also consider some useful, extra features for the GUI."
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.animation import FuncAnimation, PillowWriter
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import scipy.io as sio
from scipy.spatial import ConvexHull
import os
import sys
from pathlib import Path


# ==========================================
# 1. DATA MANAGEMENT (Strict - Left Leg)
# ==========================================

class DataManager:
    """Handles data loading and processing. Crashes if files are missing."""

    def __init__(self, data_dir="data"):
        self.dir = Path(data_dir)
        self.is_ready = False
        self.verification_status = "Not Checked"

        # Configuration: DOF Mapping for Text File Headers (SWITCHED TO LEFT)
        self.dof_map = {
            'tilt': 'pelvis_tilt',
            'hip': 'hip_flexion_l',
            'knee': 'knee_angle_l',
            'ankle': 'ankle_angle_l'
        }

        # Updated Subset List (Left Leg Specific)
        self.target_subset = [
            'iliacus_l',
            'glut_max1_l', 'glut_med1_l',
            'add_long_l',
            'bifemlh_l', 'bifemsh_l',
            'semiten_l', 'semimem_l',
            'tib_ant_l',
            'per_brev_l',
            'med_gas_l', 'lat_gas_l',
            'soleus_l', 'tib_post_l'
        ]

        self.muscle_names = []
        self.R_full = None  # Moment Arms
        self.F_so = None  # Raw Forces
        self.F_max_all = None  # Force capacities
        self.q_deg = None  # Kinematics
        self.tau_demand = None  # ID
        self.tau_verify = None

        self.L_thigh = 0.4
        self.L_shank = 0.42
        self.L_foot = 0.2

    def load_data(self):
        print("--- LOADING DATA (STRICT - LEFT LEG) ---")

        # 1. Load Moment Arms
        ma_path = self.dir / "momentArm_7x92x120.mat"
        if not ma_path.exists(): raise FileNotFoundError(f"{ma_path} missing.")
        ma_mat = sio.loadmat(str(ma_path))
        key = [k for k in ma_mat.keys() if not k.startswith('__')][0]
        self.R_full = ma_mat[key]  # Shape: [DOF, Muscles, Time=120]

        # Define Master Time Vector based on Moment Arms (Assuming 0 to 1 normalized or similar,
        # but here we just need index alignment 0..119)
        n_frames_master = self.R_full.shape[2]
        # Create a generic time vector for the master data (0 to 1)
        time_master = np.linspace(0, 1, n_frames_master)

        # 2. Load Muscle Forces
        f_path = self.dir / "force_92x120.mat"
        if not f_path.exists(): raise FileNotFoundError(f"{f_path} missing.")
        f_mat = sio.loadmat(str(f_path))
        key_f = [k for k in f_mat.keys() if 'Force' in k][0]
        self.F_so = f_mat[key_f]
        if self.F_so.shape[0] != 92:
            self.F_so = self.F_so.T

        # 3. Get Muscle Names
        sto_path = self.dir / "3DGaitModel2392_StaticOptimization_force.sto"
        if not sto_path.exists(): raise FileNotFoundError(f"{sto_path} missing.")
        _, self.muscle_names, _ = self._parse_sto(sto_path)
        self.muscle_names = self.muscle_names[:92]

        # 4. Estimate F_max
        self.F_max_all = np.max(self.F_so, axis=1)
        self.F_max_all[self.F_max_all < 100] = 100

        # 5. Load Kinematics (Left Leg Headers)
        kin_path = self.dir / "AB09_stair_s20dg_03.txt"
        if not kin_path.exists(): raise FileNotFoundError(f"{kin_path} missing.")
        self.q_deg = self._parse_stacked_txt(kin_path)

        # 6. Load Segment Lengths
        seg_path = self.dir / "segment_lengths.txt"
        if seg_path.exists():
            with open(seg_path, 'r') as f:
                vals = [float(line.strip()) for line in f if line.strip() and (line[0].isdigit() or line[0] == '.')]
            if len(vals) >= 3:
                self.L_thigh, self.L_shank, self.L_foot = vals[0], vals[1], vals[2]

        # 7. Load Inverse Dynamics (Left Leg Moments) WITH INTERPOLATION
        id_path = self.dir / "inverse_dynamics_scaled.sto"
        if not id_path.exists(): raise FileNotFoundError(f"{id_path} missing.")

        # Parse returns data, column names, AND the time vector from the file
        id_data, id_names, id_time = self._parse_sto(id_path)

        # Identify Columns
        target_cols = ['hip_flexion_l_moment', 'knee_angle_l_moment', 'ankle_angle_l_moment']
        found_indices = []
        for tgt in target_cols:
            match = next((i for i, name in enumerate(id_names) if tgt == name), None)
            if match is None:
                match = next((i for i, name in enumerate(id_names) if tgt in name), None)
            if match is not None:
                found_indices.append(match)
            else:
                print(f"Warning: ID Column '{tgt}' not found.")

        if len(found_indices) == 3:
            # Extract raw data [N_id_frames x 3]
            tau_raw = id_data[:, found_indices]

            # Interpolation Logic matching MATLAB
            # Check if Time needs interpolation (Length mismatch or value mismatch)
            # Normalizing id_time to 0..1 range to match master frames
            # (assuming both represent 0% to 100% gait cycle)
            if len(id_time) > 1:
                t_norm_id = (id_time - id_time[0]) / (id_time[-1] - id_time[0])
            else:
                t_norm_id = id_time  # Fallback

            if tau_raw.shape[0] != n_frames_master:
                print(f" Interpolating Inverse Dynamics: {tau_raw.shape[0]} frames -> {n_frames_master} frames")
                tau_interp = np.zeros((n_frames_master, 3))
                for i in range(3):
                    # interp(x_new, x_old, y_old)
                    tau_interp[:, i] = np.interp(time_master, t_norm_id, tau_raw[:, i])
                self.tau_demand = tau_interp.T  # [3 x 120]
            else:
                self.tau_demand = tau_raw.T
        else:
            print("Critical Warning: Could not map ID columns. Demand dot may be zero.")
            self.tau_demand = np.zeros((3, n_frames_master))

        # 8. Verification
        vt_path = self.dir / "jointTorque_7x120.mat"
        if vt_path.exists():
            vt_mat = sio.loadmat(str(vt_path))
            key_vt = [k for k in vt_mat.keys() if not k.startswith('__')][0]
            self.tau_verify = vt_mat[key_vt]
            self._verify_torque_calculation()
        else:
            self.verification_status = "No Ref File"

        self.is_ready = True
        print("--- DATA LOADED SUCCESSFULLY ---")

    def _verify_torque_calculation(self):
        n_dof, n_mus, n_time = self.R_full.shape
        tau_calc = np.zeros((n_dof, n_time))
        for t in range(n_time):
            tau_calc[:, t] = self.R_full[:, :, t] @ self.F_so[:, t]

        diff = tau_calc - self.tau_verify
        rmse = np.sqrt(np.mean(diff ** 2))
        self.verification_status = f"RMSE: {rmse:.4f}"

    def _parse_stacked_txt(self, filepath):
        data_dict = {}
        current_key = None
        current_vals = []

        with open(filepath, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line: continue
            if line[0].isalpha():
                if current_key: data_dict[current_key] = np.array(current_vals)
                current_key = line
                current_vals = []
            else:
                try:
                    current_vals.append(float(line))
                except ValueError:
                    pass

        if current_key: data_dict[current_key] = np.array(current_vals)

        try:
            tilt = data_dict.get(self.dof_map['tilt'], np.zeros_like(data_dict[self.dof_map['hip']]))
            hip = data_dict[self.dof_map['hip']]
            knee = data_dict[self.dof_map['knee']]
            ankle = data_dict[self.dof_map['ankle']]

            min_len = min(len(tilt), len(hip), len(knee), len(ankle))
            return np.column_stack((tilt[:min_len], hip[:min_len], knee[:min_len], ankle[:min_len]))
        except KeyError as e:
            raise ValueError(f"Missing required DOF header in file: {e}")

    def _parse_sto(self, filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        header_end = 0
        for i, line in enumerate(lines):
            if 'endheader' in line:
                header_end = i + 1
                break

        if header_end < len(lines):
            col_names = lines[header_end].strip().split()
            data = np.loadtxt(lines[header_end + 1:])
            # Return Data, Names, and Time Vector (col 0)
            return data[:, 1:], col_names[1:], data[:, 0]
        else:
            return np.array([]), [], []

    def get_muscle_indices(self, use_subset=False):
        """Filters for LEFT side muscles (_l)"""
        indices = []
        for i, name in enumerate(self.muscle_names):
            name_lower = name.lower()

            # Global Filter: Must be LEFT side (_l)
            if not name_lower.endswith('_l'):
                continue

            if use_subset:
                is_target = False
                for target in self.target_subset:
                    # Specific matching for the user provided list
                    if name_lower.startswith(target.lower()):
                        is_target = True
                        break
                if not is_target: continue

            indices.append(i)
        return sorted(indices)


# ==========================================
# 2. MATH ENGINE (Left Leg Focus)
# ==========================================

class Engine:
    def __init__(self, data_obj):
        self.d = data_obj
        self.n_samples = 2000
        self.alphas_base = np.random.rand(self.n_samples, 92)

    def get_kinematic_chain(self, frame):
        q = self.d.q_deg[frame]  # [Tilt, Hip, Knee, Ankle]

        t_pelvis = np.radians(q[0])
        t_thigh = t_pelvis + np.radians(q[1])
        t_shank = t_thigh + np.radians(q[2])
        t_foot = t_shank + np.radians(q[3]) + np.pi / 2

        l1, l2, l3 = self.d.L_thigh, self.d.L_shank, self.d.L_foot

        def polar_to_cart(L, theta):
            return np.array([L * np.sin(theta), -L * np.cos(theta)])

        p_hip = np.array([0.0, 0.0])
        p_knee = p_hip + polar_to_cart(l1, t_thigh)
        p_ankle = p_knee + polar_to_cart(l2, t_shank)
        p_toe = p_ankle + polar_to_cart(l3, t_foot)

        return {
            'p_hip': p_hip, 'p_knee': p_knee, 'p_ankle': p_ankle, 'p_toe': p_toe,
            't_thigh': t_thigh, 't_shank': t_shank, 't_foot': t_foot
        }

    def get_force_jacobian(self, frame):
        kin = self.get_kinematic_chain(frame)
        p_toe = kin['p_toe']
        p_hip = kin['p_hip']
        p_knee = kin['p_knee']
        p_ankle = kin['p_ankle']

        r_hip = p_toe - p_hip
        r_knee = p_toe - p_knee
        r_ankle = p_toe - p_ankle

        j1 = np.array([-r_hip[1], r_hip[0]])
        j2 = np.array([-r_knee[1], r_knee[0]])
        j3 = np.array([-r_ankle[1], r_ankle[0]])

        J = np.column_stack([j1, j2, j3])
        return J

    def compute_polytope(self, frame, active_indices, mode='torque'):
        if not active_indices:
            return np.zeros((1, 3))

            # INDICES UPDATE for Left Leg
        # Standard OpenSim Order assumed: 0:Tilt, 1-3:Right, 4-6:Left
        # So HipL=4, KneeL=5, AnkleL=6
        R_slice = self.d.R_full[4:7, active_indices, frame]
        F_slice = self.d.F_max_all[active_indices]
        generators = R_slice * F_slice

        if mode == 'force':
            J = self.get_force_jacobian(frame)
            try:
                J_T_pinv = np.linalg.pinv(J.T)
                generators = J_T_pinv @ generators
            except np.linalg.LinAlgError:
                return np.zeros((3, 2))

        n_gen = len(active_indices)
        alphas = self.alphas_base[:, :n_gen]
        points = alphas @ generators.T

        total_sum = np.sum(generators, axis=1)
        points = np.vstack([points, np.zeros_like(total_sum), total_sum])

        try:
            hull = ConvexHull(points)
            if mode == 'force':
                return points[hull.vertices]
            else:
                return points, hull
        except:
            return points, None


# ==========================================
# 3. GUI APPLICATION
# ==========================================

class NeuromechApp:
    def __init__(self, root, data_obj):
        self.root = root
        self.root.title("Feasible Torque/Force Sets Neuromechanics GUI")
        self.root.geometry("1400x900")

        self.d = data_obj
        self.eng = Engine(data_obj)

        # State
        self.frame = 0
        self.is_playing = False
        self.plot_mode = "torque"
        self.use_subset = True
        self.speed_var = tk.DoubleVar(value=1.0)
        self.chk_vars = {}

        self.artists = {'leg': {}, 'poly': {}, 'demand': {}}
        self.ax_poly_ref = None
        self.toolbar = None  # Reference to toolbar
        self.gs = None  # Reference to GridSpec

        self._setup_ui()
        self._init_plots()
        self.refresh_muscle_list()
        self.update_frame(0)

    def _setup_ui(self):
        main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(main_pane, width=350, padding=10)
        right_panel = ttk.Frame(main_pane, padding=10)
        main_pane.add(left_panel)
        main_pane.add(right_panel)

        gb_info = ttk.LabelFrame(left_panel, text="Model Status")
        gb_info.pack(fill="x", pady=5)
        ttk.Label(gb_info, text=self.d.verification_status).pack(anchor="w")

        gb_set = ttk.LabelFrame(left_panel, text="Visualization")
        gb_set.pack(fill="x", pady=5)

        self.var_mode = tk.StringVar(value="torque")
        ttk.Radiobutton(gb_set, text="Torque Space (3D: Hip/Knee/Ankle)",
                        variable=self.var_mode, value="torque",
                        command=self.on_mode_change).pack(anchor="w")
        ttk.Radiobutton(gb_set, text="Force Space (2D: Fx/Fy)",
                        variable=self.var_mode, value="force",
                        command=self.on_mode_change).pack(anchor="w")

        self.var_subset = tk.BooleanVar(value=True)
        ttk.Checkbutton(gb_set, text="Limit to 14 Muscles",
                        variable=self.var_subset,
                        command=self.refresh_muscle_list).pack(anchor="w", pady=5)

        ttk.Button(gb_set, text="Select All", command=self.select_all).pack(side="left", padx=2)
        ttk.Button(gb_set, text="Select None", command=self.select_none).pack(side="left", padx=2)

        # UPDATE LABEL
        lbl = ttk.Label(left_panel, text="Active Muscles (Left Only)", font=("Bold", 10))
        lbl.pack(anchor="w", pady=(10, 0))

        list_container = ttk.Frame(left_panel)
        list_container.pack(fill="both", expand=True)

        canvas = tk.Canvas(list_container, bg="#f0f0f0")
        scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=canvas.yview)
        self.scroll_frame = ttk.Frame(canvas)
        self.scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.fig = plt.Figure(figsize=(10, 6), dpi=100)

        # Use GridSpec to make Posture plot smaller (Ratio 1 : 1.8)
        self.gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1.8])
        self.ax_stick = self.fig.add_subplot(self.gs[0])

        self.canvas = FigureCanvasTkAgg(self.fig, right_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Frame for Toolbar to allow recreation
        self.toolbar_frame = ttk.Frame(right_panel)
        self.toolbar_frame.pack(fill="x", pady=2)

        # Controls Frame
        ctrl_frame = ttk.Frame(right_panel)
        ctrl_frame.pack(fill="x", pady=5)

        self.btn_play = ttk.Button(ctrl_frame, text="▶ Play", command=self.toggle_play)
        self.btn_play.pack(side="left")

        ttk.Label(ctrl_frame, text="Speed:").pack(side="left", padx=(15, 5))
        self.lbl_speed = ttk.Label(ctrl_frame, text="1.0x", width=4)
        self.lbl_speed.pack(side="left")
        scale_speed = ttk.Scale(ctrl_frame, from_=0.1, to=2.0, variable=self.speed_var,
                                command=lambda v: self.lbl_speed.config(text=f"{float(v):.1f}x"))
        scale_speed.pack(side="left", padx=5)

        self.slider = ttk.Scale(ctrl_frame, from_=0, to=119, command=self.on_slider)
        self.slider.pack(side="left", fill="x", expand=True, padx=10)
        self.lbl_frame = ttk.Label(ctrl_frame, text="0", width=5)
        self.lbl_frame.pack(side="left")

        ttk.Button(ctrl_frame, text="Export GIF", command=self.export_gif).pack(side="right")

    def _create_toolbar(self):
        """Recreates the toolbar to ensure it tracks correct axes"""
        if self.toolbar:
            self.toolbar.destroy()

        # Create new toolbar in the container frame
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

    def _init_plots(self):
        self.ax_stick.set_xlim(-0.5, 1.0)
        self.ax_stick.set_ylim(-1.2, 0.5)
        self.ax_stick.set_aspect('equal')
        self.ax_stick.grid(True)
        self.ax_stick.set_title("Kinematics (Left Leg) (m)")

        self.artists['leg']['thigh'], = self.ax_stick.plot([], [], 'o-', lw=5, color='#333333', label='Thigh')
        self.artists['leg']['shank'], = self.ax_stick.plot([], [], 'o-', lw=5, color='#0066cc', label='Shank')
        self.artists['leg']['foot'], = self.ax_stick.plot([], [], 'o-', lw=5, color='#009933', label='Foot')
        self.artists['leg']['ground'] = self.ax_stick.axhline(-1.0, color='gray', lw=2)

        self._reset_poly_axes()

    def _reset_poly_axes(self):
        if self.ax_poly_ref:
            self.fig.delaxes(self.ax_poly_ref)

        mode = self.var_mode.get()

        if mode == 'torque':
            self.ax_poly_ref = self.fig.add_subplot(self.gs[1], projection='3d')
            self.ax_poly_ref.set_xlabel("Hip (Nm)")
            self.ax_poly_ref.set_ylabel("Knee (Nm)")
            self.ax_poly_ref.set_zlabel("Ankle (Nm)")
            self.ax_poly_ref.set_title("Feasible Torque Set (3D)")
            lim = 300
            self.ax_poly_ref.set_xlim(-lim, lim)
            self.ax_poly_ref.set_ylim(-lim, lim)
            self.ax_poly_ref.set_zlim(-lim, lim)

            self.artists['poly']['verts_3d'] = self.ax_poly_ref.scatter([], [], [], c='blue', s=10, depthshade=False)
            self.artists['demand']['ray_3d'], = self.ax_poly_ref.plot([], [], [], 'r--', lw=1.5, label='Demand')
            self.artists['demand']['dot_3d'], = self.ax_poly_ref.plot([], [], [], 'ro', ms=6)
            self.artists['poly']['edges_3d'] = None

        else:
            self.ax_poly_ref = self.fig.add_subplot(self.gs[1])
            self.ax_poly_ref.set_xlabel("Fx (N)")
            self.ax_poly_ref.set_ylabel("Fy (N)")
            self.ax_poly_ref.set_title("Feasible Force Set (2D)")
            self.ax_poly_ref.grid(True)
            self.ax_poly_ref.axhline(0, color='k', lw=0.5)
            self.ax_poly_ref.axvline(0, color='k', lw=0.5)
            lim = 2000
            self.ax_poly_ref.set_xlim(-lim, lim)
            self.ax_poly_ref.set_ylim(-lim, lim)

            self.artists['poly']['patch'] = Polygon([[0, 0]], closed=True, fc='cornflowerblue', alpha=0.4, ec='blue')
            self.ax_poly_ref.add_patch(self.artists['poly']['patch'])
            self.artists['poly']['verts_2d'], = self.ax_poly_ref.plot([], [], 'b.', ms=4)
            self.artists['demand']['dot'], = self.ax_poly_ref.plot([], [], 'ro', ms=8, zorder=5)
            self.artists['demand']['ray'], = self.ax_poly_ref.plot([], [], 'r--', alpha=0.5, lw=1.5)

        # RECREATE TOOLBAR so the Home button knows about the new axes
        self._create_toolbar()

        self.fig.tight_layout()
        self.canvas.draw()

    def refresh_muscle_list(self):
        for widget in self.scroll_frame.winfo_children(): widget.destroy()
        self.chk_vars = {}
        indices = self.d.get_muscle_indices(self.var_subset.get())
        groups = {
            "Hip / Pelvis": ["glut", "ili", "pso", "gem", "pir", "add", "pect", "quad_fem", "obt"],
            "Knee / Thigh": ["vas", "rec", "bic", "sem", "gra", "sar", "ten_fas"],
            "Ankle / Foot": ["tib", "gas", "sol", "per", "fle", "ext"],
        }
        sorted_groups = {k: [] for k in groups};
        sorted_groups["Other"] = []
        for idx in indices:
            name = self.d.muscle_names[idx].lower()
            placed = False
            for group, keywords in groups.items():
                if any(k in name for k in keywords):
                    sorted_groups[group].append(idx);
                    placed = True;
                    break
            if not placed: sorted_groups["Other"].append(idx)
        for group_name, group_indices in sorted_groups.items():
            if not group_indices: continue
            ttk.Label(self.scroll_frame, text=f"--- {group_name} ---", font=("Arial", 9, "bold")).pack(anchor="w",
                                                                                                       pady=(5, 2))
            for idx in group_indices:
                name = self.d.muscle_names[idx]
                var = tk.BooleanVar(value=True)
                self.chk_vars[idx] = var
                cb = ttk.Checkbutton(self.scroll_frame, text=name, variable=var, command=self.on_param_change)
                cb.pack(anchor="w", padx=15)
        self.on_param_change()

    def select_all(self):
        for v in self.chk_vars.values(): v.set(True)
        self.on_param_change()

    def select_none(self):
        for v in self.chk_vars.values(): v.set(False)
        self.on_param_change()

    def on_mode_change(self):
        self.plot_mode = self.var_mode.get()
        self._reset_poly_axes()
        self.update_frame(self.frame)

    def on_param_change(self):
        self.update_frame(self.frame)

    def on_slider(self, val):
        self.frame = int(float(val))
        self.update_frame(self.frame)

    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.btn_play.config(text="|| Pause" if self.is_playing else "▶ Play")
        if self.is_playing: self.animate()

    def animate(self):
        if not self.is_playing: return
        self.frame = (self.frame + 1) % 120
        self.slider.set(self.frame)
        speed = self.speed_var.get()
        delay = int(40 / speed) if speed > 0 else 40
        self.root.after(delay, self.animate)

    def update_frame(self, f):
        self.lbl_frame.config(text=str(f))

        # 1. Kinematics (Updated Hierarchical Logic)
        kin = self.eng.get_kinematic_chain(f)
        p_hip = kin['p_hip']
        p_knee = kin['p_knee']
        p_ankle = kin['p_ankle']
        p_toe = kin['p_toe']

        self.artists['leg']['thigh'].set_data([p_hip[0], p_knee[0]], [p_hip[1], p_knee[1]])
        self.artists['leg']['shank'].set_data([p_knee[0], p_ankle[0]], [p_knee[1], p_ankle[1]])
        self.artists['leg']['foot'].set_data([p_ankle[0], p_toe[0]], [p_ankle[1], p_toe[1]])

        # 2. Polytope
        active_indices = [idx for idx, v in self.chk_vars.items() if v.get()]
        res = self.eng.compute_polytope(f, active_indices, self.plot_mode)

        # Demand (Torque)
        tau = self.d.tau_demand[:, f]  # Hip, Knee, Ankle

        if self.plot_mode == 'torque':
            points, hull = res

            if hull is not None:
                verts = points[hull.vertices]
                self.artists['poly']['verts_3d']._offsets3d = (verts[:, 0], verts[:, 1], verts[:, 2])

                if self.artists['poly']['edges_3d']:
                    self.artists['poly']['edges_3d'].remove()

                edges = []
                for s in hull.simplices:
                    s = np.append(s, s[0])
                    edges.append(points[s])

                edge_col = Poly3DCollection(edges, alpha=0.1, facecolor='cyan', edgecolor='blue')
                self.artists['poly']['edges_3d'] = self.ax_poly_ref.add_collection3d(edge_col)
            else:
                self.artists['poly']['verts_3d']._offsets3d = ([], [], [])

            self.artists['demand']['ray_3d'].set_data_3d([0, tau[0]], [0, tau[1]], [0, tau[2]])
            self.artists['demand']['dot_3d'].set_data_3d([tau[0]], [tau[1]], [tau[2]])

        else:  # Force Mode
            verts = res
            if len(verts) > 0:
                self.artists['poly']['patch'].set_xy(verts)
                self.artists['poly']['verts_2d'].set_data(verts[:, 0], verts[:, 1])

            # Map Torque -> Force using Correct Jacobian
            J = self.eng.get_force_jacobian(f)
            try:
                # F = (J^T)+ * tau
                f_demand = np.linalg.pinv(J.T) @ tau
                dx, dy = f_demand[0], f_demand[1]
            except:
                dx, dy = 0, 0

            self.artists['demand']['dot'].set_data([dx], [dy])
            self.artists['demand']['ray'].set_data([0, dx], [0, dy])

        self.canvas.draw_idle()

    def export_gif(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".gif", filetypes=[("GIF", "*.gif")])
        if not file_path: return
        self.is_playing = False
        messagebox.showinfo("Export", "Generating GIF... Please wait (UI will freeze).")

        def update_anim(frame_idx):
            self.frame = frame_idx
            self.update_frame(frame_idx)

        anim = FuncAnimation(self.fig, update_anim, frames=range(0, 120, 2), blit=False)
        try:
            anim.save(file_path, writer=PillowWriter(fps=15))
            messagebox.showinfo("Success", "GIF Saved Successfully!")
        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    if not os.path.exists("data"):
        print("Error: 'data' folder not found.")
        sys.exit(1)
    root = tk.Tk()
    dm = DataManager("data")
    try:
        dm.load_data()
        if dm.is_ready:
            app = NeuromechApp(root, dm)
            root.mainloop()
    except Exception as e:
        messagebox.showerror("Crash", f"Critical Error Loading Data:\n{str(e)}")
        raise e
