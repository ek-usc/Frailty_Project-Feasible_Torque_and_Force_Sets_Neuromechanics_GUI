# Updated data files:
# (none)
#
# Placeholder data files:
#   muscle_data:
#       muscle_parameters.csv
#       muscle_forces.csv
#       activation_timeseries.csv
#   moment_arms:
#       moment_arm_matrix.csv
#   kinematics:
#       joint_angles.csv
#       segment_lengths.json
#   force_polytopes:
#       polytopes.npz
#   config.yaml


"""
Lower Limb Force Capacity Visualization GUI
BME 504 Final Project - Stair Climbing Analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import yaml
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
import sys
from pathlib import Path
import os


class DataLoader:
    """Load and validate all input data files"""

    def __init__(self, data_path="data"):
        self.data_path = Path(data_path)
        self.muscles = None
        self.kinematics = None
        self.moment_arms = None
        self.activations = None
        self.forces = None
        self.segment_lengths = None
        self.config = None

    def check_files_exist(self):
        """Check if all required files exist"""
        required_files = [
            self.data_path / "muscle_data" / "muscle_parameters.csv",
            self.data_path / "kinematics" / "joint_angles.csv",
            self.data_path / "kinematics" / "segment_lengths.json",
            self.data_path / "moment_arms" / "moment_arm_matrix.csv",
            self.data_path / "muscle_data" / "activation_timeseries.csv",
            self.data_path / "muscle_data" / "muscle_forces.csv",
            self.data_path / "config.yaml"
        ]

        missing_files = []
        for file_path in required_files:
            if not file_path.exists() or file_path.stat().st_size == 0:
                missing_files.append(str(file_path))

        return len(missing_files) == 0, missing_files

    def load_all(self):
        """Load all required data files"""
        print("Loading data files...")

        # Load muscle parameters
        muscle_file = self.data_path / "muscle_data" / "muscle_parameters.csv"
        self.muscles = pd.read_csv(muscle_file, index_col='muscle_name')
        print(f"✓ Loaded {len(self.muscles)} muscles")

        # Load kinematics
        joint_file = self.data_path / "kinematics" / "joint_angles.csv"
        self.kinematics = pd.read_csv(joint_file)
        print(f"✓ Loaded kinematics: {len(self.kinematics)} time points")

        # Load segment lengths
        segment_file = self.data_path / "kinematics" / "segment_lengths.json"
        try:
            with open(segment_file, 'r') as f:
                content = f.read()
                if not content:
                    raise ValueError("Segment lengths file is empty")
                self.segment_lengths = json.loads(content)
            print(f"✓ Loaded segment lengths")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"✗ Error loading segment lengths: {e}")
            raise

        # Load moment arms
        ma_file = self.data_path / "moment_arms" / "moment_arm_matrix.csv"
        self.moment_arms = pd.read_csv(ma_file)
        print(f"✓ Loaded moment arms")

        # Load activations
        act_file = self.data_path / "muscle_data" / "activation_timeseries.csv"
        self.activations = pd.read_csv(act_file)
        print(f"✓ Loaded muscle activations")

        # Load forces
        force_file = self.data_path / "muscle_data" / "muscle_forces.csv"
        self.forces = pd.read_csv(force_file)
        print(f"✓ Loaded muscle forces")

        # Load config
        config_file = self.data_path / "config.yaml"
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        print(f"✓ Loaded configuration")

        self.validate_data()

    def validate_data(self):
        """Validate data consistency"""
        n_time_points = len(self.kinematics)

        assert len(self.activations) == n_time_points, \
            f"Activation data length mismatch: {len(self.activations)} vs {n_time_points}"
        assert len(self.forces) == n_time_points, \
            f"Force data length mismatch: {len(self.forces)} vs {n_time_points}"

        # Check muscle names consistency
        muscle_names = self.muscles.index.tolist()
        act_muscles = [col for col in self.activations.columns if col != 'time']

        missing_in_act = set(muscle_names) - set(act_muscles)
        if missing_in_act:
            print(f"Warning: Muscles in parameters but not in activations: {missing_in_act}")

        print("✓ Data validation passed")


class LegModel:
    """Forward kinematics and visualization for 2D leg model"""

    def __init__(self, segment_lengths):
        self.thigh_length = segment_lengths['thigh_length']
        self.shank_length = segment_lengths['shank_length']
        self.foot_length = segment_lengths['foot_length']

    def compute_positions(self, hip_angle, knee_angle, ankle_angle):
        """
        Compute joint positions given angles (in degrees)
        Returns: hip, knee, ankle, toe positions as (x, y) tuples
        Origin at hip joint
        """
        # Convert to radians
        hip_rad = np.radians(hip_angle)
        knee_rad = np.radians(knee_angle)
        ankle_rad = np.radians(ankle_angle)

        # Hip position (origin)
        hip = np.array([0.0, 0.0])

        # Knee position
        knee_x = self.thigh_length * np.sin(hip_rad)
        knee_y = -self.thigh_length * np.cos(hip_rad)
        knee = np.array([knee_x, knee_y])

        # Ankle position
        # Knee angle is relative to thigh
        shank_angle = hip_rad + knee_rad
        ankle_x = knee_x + self.shank_length * np.sin(shank_angle)
        ankle_y = knee_y - self.shank_length * np.cos(shank_angle)
        ankle = np.array([ankle_x, ankle_y])

        # Toe position
        # Ankle angle is relative to shank
        foot_angle = shank_angle + ankle_rad
        toe_x = ankle_x + self.foot_length * np.cos(foot_angle)
        toe_y = ankle_y + self.foot_length * np.sin(foot_angle)
        toe = np.array([toe_x, toe_y])

        return hip, knee, ankle, toe


class PolytopeComputer:
    """Compute feasible force polytopes"""

    def __init__(self, muscle_params, segment_lengths):
        self.muscle_params = muscle_params
        self.body_weight = segment_lengths['subject_mass'] * 9.81

    def compute_jacobian(self, hip_angle, knee_angle, ankle_angle,
                        thigh_length, shank_length):
        """
        Compute Jacobian matrix J: maps joint torques to endpoint forces
        J is 2x3 (Fx, Fy) = J^-T * (tau_hip, tau_knee, tau_ankle)

        Simplified 2D planar model
        """
        # Convert to radians
        theta1 = np.radians(hip_angle)
        theta2 = np.radians(knee_angle)

        # Simplified Jacobian for 2D leg
        l1 = thigh_length
        l2 = shank_length

        J = np.array([
            [-l1*np.cos(theta1) - l2*np.cos(theta1+theta2),
             -l2*np.cos(theta1+theta2)],
            [-l1*np.sin(theta1) - l2*np.sin(theta1+theta2),
             -l2*np.sin(theta1+theta2)]
        ])

        # Add ankle column (simplified)
        J = np.column_stack([J, np.array([-0.1*np.cos(theta1+theta2),
                                          -0.1*np.sin(theta1+theta2)])])

        return J

    def compute_polytope(self, moment_arms, muscle_forces, jacobian,
                        selected_muscles):
        """
        Compute force polytope vertices

        Args:
            moment_arms: dict with {muscle_name: [hip_ma, knee_ma, ankle_ma]}
            muscle_forces: dict with {muscle_name: max_force}
            jacobian: 2x3 or 3x3 matrix
            selected_muscles: list of muscle names to include

        Returns:
            vertices: Nx2 array of (Fx, Fy) points defining polytope boundary
        """
        # Build moment arm matrix R (3 joints x N muscles)
        R = []
        F_max = []

        for muscle in selected_muscles:
            if muscle in moment_arms and muscle in muscle_forces:
                R.append(moment_arms[muscle])
                F_max.append(muscle_forces[muscle])

        if len(R) == 0:
            return np.array([[0, 0]])

        R = np.array(R).T  # Shape: (3, n_muscles)
        F_max = np.array(F_max)

        # Sample muscle activation space
        n_muscles = len(selected_muscles)

        if n_muscles <= 10:
            # For small number of muscles, use all vertices
            from itertools import product
            activations = np.array(list(product([0, 1], repeat=n_muscles)))
        else:
            # For many muscles, sample randomly
            n_samples = min(2**n_muscles, 1000)
            activations = np.random.rand(n_samples, n_muscles)
            # Add corner points
            corners = np.eye(n_muscles)
            activations = np.vstack([activations, corners, np.zeros((1, n_muscles))])

        # Compute endpoint forces for each activation pattern
        endpoint_forces = []

        for a in activations:
            # Muscle forces
            F_muscle = a * F_max

            # Joint torques: tau = R * F_muscle
            tau = R @ F_muscle

            # Endpoint force: F = J^-T * tau
            try:
                if jacobian.shape[0] == 2:
                    # Use only hip and knee torques for 2D force
                    J_inv_T = np.linalg.pinv(jacobian[:, :2]).T
                    F_endpoint = J_inv_T @ tau[:2]
                else:
                    J_inv_T = np.linalg.pinv(jacobian).T
                    F_endpoint = J_inv_T @ tau
                    F_endpoint = F_endpoint[:2]  # Take only Fx, Fy

                endpoint_forces.append(F_endpoint)
            except:
                continue

        if len(endpoint_forces) == 0:
            return np.array([[0, 0]])

        endpoint_forces = np.array(endpoint_forces)

        # Compute convex hull
        try:
            hull = ConvexHull(endpoint_forces)
            vertices = endpoint_forces[hull.vertices]
        except:
            vertices = endpoint_forces

        return vertices


class StairClimbingGUI:
    """Main GUI application"""

    def __init__(self, root, data_loader):
        self.root = root
        self.root.title("Lower Limb Force Capacity - Stair Climbing Analysis")

        self.data = data_loader
        self.leg_model = LegModel(data_loader.segment_lengths)
        self.polytope_computer = PolytopeComputer(
            data_loader.muscles,
            data_loader.segment_lengths
        )

        # State variables
        self.current_frame = 0
        self.n_frames = len(data_loader.kinematics)
        self.is_playing = False
        self.is_exporting = False
        self.animation = None

        # Selected muscles (default: all)
        self.muscle_names = data_loader.muscles.index.tolist()
        self.muscle_vars = {}  # Will hold tk.BooleanVar for each muscle

        # Default view limits (zoomed out 0.5x = 2x range, panned down 500 N)
        self.default_leg_xlim = [-0.6, 0.6]
        self.default_leg_ylim = [-1.2, 0.2]
        self.default_polytope_xlim = [-1000, 1000]
        self.default_polytope_ylim = [-1800, 2600]

        # Current view limits (will be updated by user pan/zoom)
        self.current_leg_xlim = self.default_leg_xlim.copy()
        self.current_leg_ylim = self.default_leg_ylim.copy()
        self.current_polytope_xlim = self.default_polytope_xlim.copy()
        self.current_polytope_ylim = self.default_polytope_ylim.copy()

        # Pre-compute moment arms interpolation
        self.setup_moment_arms()

        # Create GUI layout
        self.create_gui()

        # Initial draw
        self.update_visualization()

    def setup_moment_arms(self):
        """Pre-process moment arms for fast lookup"""
        self.moment_arm_dict = {}

        for muscle in self.muscle_names:
            muscle_data = self.data.moment_arms[
                self.data.moment_arms['muscle'] == muscle
            ]

            if len(muscle_data) > 0:
                times = muscle_data['time'].values
                hip_ma = muscle_data['hip_moment_arm'].values
                knee_ma = muscle_data['knee_moment_arm'].values
                ankle_ma = muscle_data['ankle_moment_arm'].values

                # Create interpolation functions
                if len(times) > 1:
                    self.moment_arm_dict[muscle] = {
                        'hip': interp1d(times, hip_ma, fill_value='extrapolate'),
                        'knee': interp1d(times, knee_ma, fill_value='extrapolate'),
                        'ankle': interp1d(times, ankle_ma, fill_value='extrapolate'),
                    }
                else:
                    # Single time point - use constant value
                    self.moment_arm_dict[muscle] = {
                        'hip': lambda t: hip_ma[0],
                        'knee': lambda t: knee_ma[0],
                        'ankle': lambda t: ankle_ma[0],
                    }

    def get_moment_arms(self, time_idx):
        """Get moment arms at specific time index"""
        time = self.data.kinematics.iloc[time_idx]['time']
        ma_dict = {}

        for muscle in self.muscle_names:
            if muscle in self.moment_arm_dict:
                try:
                    ma_dict[muscle] = [
                        float(self.moment_arm_dict[muscle]['hip'](time)),
                        float(self.moment_arm_dict[muscle]['knee'](time)),
                        float(self.moment_arm_dict[muscle]['ankle'](time))
                    ]
                except:
                    ma_dict[muscle] = [0.0, 0.0, 0.0]
            else:
                ma_dict[muscle] = [0.0, 0.0, 0.0]

        return ma_dict

    def create_gui(self):
        """Create main GUI layout"""

        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel: Controls
        control_frame = ttk.Frame(main_frame, width=280)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        # Right panel: Visualization
        viz_frame = ttk.Frame(main_frame)
        viz_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # === CONTROL PANEL ===

        # Title
        title_label = ttk.Label(control_frame, text="Control Panel",
                               font=('Arial', 12, 'bold'))
        title_label.pack(pady=(0, 10))

        # Muscle selection
        muscle_frame = ttk.LabelFrame(control_frame, text="Muscle Selection",
                                      padding=10)
        muscle_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Scrollable muscle list
        canvas = tk.Canvas(muscle_frame, height=200)
        scrollbar = ttk.Scrollbar(muscle_frame, orient="vertical",
                                  command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Create checkboxes for each muscle in 2 columns
        muscle_groups = self.data.config['muscles']['groups']

        row = 0
        for group_name, muscles in muscle_groups.items():
            # Group label spanning both columns
            group_label = ttk.Label(scrollable_frame, text=f"▼ {group_name}",
                                   font=('Arial', 9, 'bold'))
            group_label.grid(row=row, column=0, columnspan=2, sticky='w', pady=(5, 2))
            row += 1

            # Create checkboxes in 2 columns
            col = 0
            for idx, muscle in enumerate(muscles):
                if muscle in self.muscle_names:
                    var = tk.BooleanVar(value=True)
                    self.muscle_vars[muscle] = var

                    cb = ttk.Checkbutton(
                        scrollable_frame,
                        text=muscle.replace('_', ' ').title(),
                        variable=var,
                        command=self.update_visualization
                    )
                    cb.grid(row=row, column=col, sticky='w', padx=(15 if col == 0 else 5, 5))

                    col += 1
                    if col >= 2:
                        col = 0
                        row += 1

            # Move to next row if we ended on column 1
            if col == 1:
                row += 1

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Select/Deselect buttons
        btn_frame = ttk.Frame(muscle_frame)
        btn_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(btn_frame, text="All",
                  command=self.select_all_muscles).pack(side=tk.LEFT,
                                                        padx=(0, 5))
        ttk.Button(btn_frame, text="None",
                  command=self.deselect_all_muscles).pack(side=tk.LEFT)

        # Time slider
        slider_frame = ttk.LabelFrame(control_frame, text="Posture", padding=10)
        slider_frame.pack(fill=tk.X, pady=(0, 10))

        self.time_label = ttk.Label(slider_frame,
                                    text=f"Time: 0.000 s (0%)")
        self.time_label.pack()

        self.phase_label = ttk.Label(slider_frame, text="Phase: Heel Strike",
                                     font=('Arial', 9, 'italic'))
        self.phase_label.pack()

        self.slider = ttk.Scale(slider_frame, from_=0, to=self.n_frames-1,
                               orient=tk.HORIZONTAL, command=self.on_slider_change)
        self.slider.pack(fill=tk.X, pady=(5, 0))

        # Playback controls
        playback_frame = ttk.LabelFrame(control_frame, text="Playback",
                                        padding=10)
        playback_frame.pack(fill=tk.X, pady=(0, 10))

        btn_container = ttk.Frame(playback_frame)
        btn_container.pack()

        self.play_btn = ttk.Button(btn_container, text="▶ Play",
                                   command=self.toggle_play, width=8)
        self.play_btn.grid(row=0, column=0, padx=2)

        ttk.Button(btn_container, text="⏹ Stop",
                  command=self.stop_animation, width=8).grid(row=0, column=1,
                                                             padx=2)

        # Speed control
        speed_frame = ttk.Frame(playback_frame)
        speed_frame.pack(pady=(5, 0))

        ttk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT)
        self.speed_var = tk.StringVar(value="1x")
        speed_combo = ttk.Combobox(speed_frame, textvariable=self.speed_var,
                                   values=["0.25x", "0.5x", "1x", "2x", "4x"],
                                   width=8, state='readonly')
        speed_combo.pack(side=tk.LEFT, padx=(5, 0))
        speed_combo.set("1x")

        # Export controls
        export_frame = ttk.LabelFrame(control_frame, text="Export", padding=10)
        export_frame.pack(fill=tk.X, pady=(0, 10))

        self.export_btn = ttk.Button(export_frame, text="📹 Export Video",
                                     command=self.export_video)
        self.export_btn.pack(fill=tk.X)

        self.export_status = ttk.Label(export_frame, text="",
                                       font=('Arial', 8, 'italic'))
        self.export_status.pack(pady=(5, 0))

        # Info display
        info_frame = ttk.LabelFrame(control_frame, text="Current State",
                                    padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True)

        self.info_text = tk.Text(info_frame, height=6, width=30,
                                font=('Courier', 8))
        self.info_text.pack(fill=tk.BOTH, expand=True)

        # === VISUALIZATION PANEL ===

        # Create matplotlib figure
        self.fig = plt.figure(figsize=(12, 7))

        # Left subplot: Leg skeleton
        self.ax_leg = self.fig.add_subplot(121)
        self.ax_leg.set_aspect('equal')
        self.ax_leg.set_xlim(self.default_leg_xlim)
        self.ax_leg.set_ylim(self.default_leg_ylim)
        self.ax_leg.set_xlabel('X (m)')
        self.ax_leg.set_ylabel('Y (m)')
        self.ax_leg.set_title('Leg Posture (Sagittal Plane)')
        self.ax_leg.grid(True, alpha=0.3)

        # Right subplot: Force polytope
        self.ax_polytope = self.fig.add_subplot(122)
        self.ax_polytope.set_aspect('equal')
        self.ax_polytope.set_xlim(self.default_polytope_xlim)
        self.ax_polytope.set_ylim(self.default_polytope_ylim)
        self.ax_polytope.set_xlabel('Horizontal Force Fx (N)')
        self.ax_polytope.set_ylabel('Vertical Force Fy (N)')
        self.ax_polytope.set_title('Feasible Force Polytope')
        self.ax_polytope.grid(True, alpha=0.3)
        self.ax_polytope.axhline(y=0, color='k', linewidth=0.5)
        self.ax_polytope.axvline(x=0, color='k', linewidth=0.5)

        plt.tight_layout()

        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add navigation toolbar for pan/zoom
        toolbar_frame = ttk.Frame(viz_frame)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

    def select_all_muscles(self):
        """Select all muscles"""
        for var in self.muscle_vars.values():
            var.set(True)
        self.update_visualization()

    def deselect_all_muscles(self):
        """Deselect all muscles"""
        for var in self.muscle_vars.values():
            var.set(False)
        self.update_visualization()

    def on_slider_change(self, value):
        """Handle slider movement"""
        self.current_frame = int(float(value))
        self.update_visualization()

    def toggle_play(self):
        """Toggle play/pause"""
        if self.is_playing:
            self.pause_animation()
        else:
            self.play_animation()

    def play_animation(self):
        """Start animation"""
        self.is_playing = True
        self.play_btn.config(text="⏸ Pause")

        speed = float(self.speed_var.get().replace('x', ''))
        interval = int(1000 / (30 * speed))  # 30 fps base

        def update_frame():
            if self.is_playing:
                self.current_frame = (self.current_frame + 1) % self.n_frames
                self.slider.set(self.current_frame)
                self.update_visualization()
                self.root.after(interval, update_frame)

        update_frame()

    def pause_animation(self):
        """Pause animation"""
        self.is_playing = False
        self.play_btn.config(text="▶ Play")

    def stop_animation(self):
        """Stop animation and reset"""
        self.pause_animation()
        self.current_frame = 0
        self.slider.set(0)
        self.update_visualization()

    def export_video(self):
        """Export animation as video file"""
        if self.is_exporting:
            return

        # Ask user for save location
        filename = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[
                ("MP4 Video", "*.mp4"),
                ("GIF Animation", "*.gif"),
                ("All Files", "*.*")
            ],
            initialfile="stair_climbing_animation.mp4"
        )

        if not filename:
            return  # User cancelled

        # Disable controls during export
        self.is_exporting = True
        self.export_btn.config(state='disabled')
        self.export_status.config(text="Exporting... Please wait")
        self.root.update()

        try:
            # Determine file format
            file_ext = Path(filename).suffix.lower()

            # Get selected muscles (save state)
            selected_muscles = [m for m, var in self.muscle_vars.items()
                              if var.get()]

            # Create a new figure for export (to avoid disrupting GUI)
            export_fig = plt.figure(figsize=(12, 7))
            ax_leg_export = export_fig.add_subplot(121)
            ax_polytope_export = export_fig.add_subplot(122)

            # Animation function
            def animate(frame):
                # Update status
                progress = int(100 * frame / self.n_frames)
                self.export_status.config(text=f"Exporting... {progress}%")
                self.root.update()

                # Clear axes
                ax_leg_export.clear()
                ax_polytope_export.clear()

                # Get frame data
                row = self.data.kinematics.iloc[frame]
                time = row['time']
                hip_angle = row['hip_angle']
                knee_angle = row['knee_angle']
                ankle_angle = row['ankle_angle']
                phase = row['phase']

                # === LEG PLOT ===
                ax_leg_export.set_aspect('equal')
                ax_leg_export.set_xlim(self.current_leg_xlim)
                ax_leg_export.set_ylim(self.current_leg_ylim)
                ax_leg_export.set_xlabel('X (m)')
                ax_leg_export.set_ylabel('Y (m)')
                ax_leg_export.set_title('Leg Posture (Sagittal Plane)')
                ax_leg_export.grid(True, alpha=0.3)

                # Compute joint positions
                hip, knee, ankle, toe = self.leg_model.compute_positions(
                    hip_angle, knee_angle, ankle_angle
                )

                # Draw ground
                ax_leg_export.plot([-0.8, 0.8], [-1.0, -1.0], 'k-', linewidth=2)
                ax_leg_export.fill_between([-0.8, 0.8], [-1.0, -1.0], [-1.3, -1.3],
                                          color='gray', alpha=0.3)

                # Draw skeleton
                ax_leg_export.plot([hip[0], knee[0]], [hip[1], knee[1]],
                                  'o-', color='#2E4057', linewidth=4, markersize=8)
                ax_leg_export.plot([knee[0], ankle[0]], [knee[1], ankle[1]],
                                  'o-', color='#048A81', linewidth=4, markersize=8)
                ax_leg_export.plot([ankle[0], toe[0]], [ankle[1], toe[1]],
                                  'o-', color='#54C6EB', linewidth=4, markersize=8)

                # Draw muscles
                activations = self.data.activations.iloc[frame]
                for muscle in selected_muscles:
                    if muscle in activations:
                        activation = activations[muscle]

                        if 'glut' in muscle or 'iliacus' in muscle:
                            start = hip + np.array([-0.05, 0.05])
                            end = (hip + knee) / 2
                        elif 'vast' in muscle or 'rectus' in muscle:
                            start = (hip + knee) / 2
                            end = knee
                        elif 'gastroc' in muscle or 'soleus' in muscle:
                            start = knee
                            end = ankle
                        else:
                            continue

                        alpha = 0.3 + 0.7 * activation
                        linewidth = 1 + 3 * activation
                        ax_leg_export.plot([start[0], end[0]], [start[1], end[1]],
                                          color='red', alpha=alpha, linewidth=linewidth)

                # Labels
                ax_leg_export.text(hip[0]-0.1, hip[1]+0.05, 'Hip', fontsize=9)
                ax_leg_export.text(knee[0]-0.1, knee[1]+0.05, 'Knee', fontsize=9)
                ax_leg_export.text(ankle[0]-0.1, ankle[1]+0.05, 'Ankle', fontsize=9)

                # Add time annotation
                ax_leg_export.text(0.02, 0.98, f"Time: {time:.3f} s\nPhase: {phase}",
                                  transform=ax_leg_export.transAxes,
                                  verticalalignment='top',
                                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                # === POLYTOPE PLOT ===
                ax_polytope_export.set_aspect('equal')
                ax_polytope_export.set_xlim(self.current_polytope_xlim)
                ax_polytope_export.set_ylim(self.current_polytope_ylim)
                ax_polytope_export.set_xlabel('Horizontal Force Fx (N)')
                ax_polytope_export.set_ylabel('Vertical Force Fy (N)')
                ax_polytope_export.set_title('Feasible Force Polytope')
                ax_polytope_export.grid(True, alpha=0.3)
                ax_polytope_export.axhline(y=0, color='k', linewidth=0.5)
                ax_polytope_export.axvline(x=0, color='k', linewidth=0.5)

                if len(selected_muscles) > 0:
                    # Get moment arms
                    moment_arms = self.get_moment_arms(frame)

                    # Get max forces
                    muscle_forces = {}
                    for muscle in selected_muscles:
                        muscle_forces[muscle] = self.data.muscles.loc[muscle, 'F0_max']

                    # Compute Jacobian
                    jacobian = self.polytope_computer.compute_jacobian(
                        hip_angle, knee_angle, ankle_angle,
                        self.leg_model.thigh_length,
                        self.leg_model.shank_length
                    )

                    # Compute polytope
                    vertices = self.polytope_computer.compute_polytope(
                        moment_arms, muscle_forces, jacobian, selected_muscles
                    )

                    # Plot polytope
                    if len(vertices) > 2:
                        polygon = Polygon(vertices, alpha=0.3, facecolor='blue',
                                        edgecolor='blue', linewidth=2)
                        ax_polytope_export.add_patch(polygon)
                        ax_polytope_export.plot(vertices[:, 0], vertices[:, 1],
                                              'bo', markersize=4)

                    # Frailty threshold
                    body_weight = self.data.segment_lengths['subject_mass'] * 9.81
                    threshold = body_weight * self.data.config['frailty']['threshold_multiplier']

                    ax_polytope_export.axhline(y=threshold, color='red',
                                             linestyle='--', linewidth=2,
                                             label=f'Frailty Threshold ({threshold:.0f} N)')
                    ax_polytope_export.legend()

                plt.tight_layout()

            # Create animation
            anim = FuncAnimation(export_fig, animate, frames=self.n_frames,
                               interval=33, repeat=False)

            # Save animation
            if file_ext == '.gif':
                writer = PillowWriter(fps=30)
                anim.save(filename, writer=writer)
            else:  # .mp4 or others
                try:
                    writer = FFMpegWriter(fps=30, bitrate=2000)
                    anim.save(filename, writer=writer)
                except Exception as e:
                    # Fallback to Pillow if FFmpeg not available
                    messagebox.showwarning(
                        "FFmpeg Not Available",
                        "FFmpeg not found. Saving as GIF instead.\n" +
                        "Install FFmpeg for MP4 export."
                    )
                    gif_filename = str(Path(filename).with_suffix('.gif'))
                    writer = PillowWriter(fps=30)
                    anim.save(gif_filename, writer=writer)
                    filename = gif_filename

            plt.close(export_fig)

            # Success message
            self.export_status.config(text="Export complete!")
            messagebox.showinfo("Export Complete",
                              f"Video saved to:\n{filename}")

        except Exception as e:
            messagebox.showerror("Export Error",
                               f"Failed to export video:\n{str(e)}")
            self.export_status.config(text="Export failed")

        finally:
            # Re-enable controls
            self.is_exporting = False
            self.export_btn.config(state='normal')
            # Clear status after 3 seconds
            self.root.after(3000, lambda: self.export_status.config(text=""))

    def update_visualization(self):
        """Update all visualizations based on current state"""

        # Save current view limits before clearing
        self.current_leg_xlim = list(self.ax_leg.get_xlim())
        self.current_leg_ylim = list(self.ax_leg.get_ylim())
        self.current_polytope_xlim = list(self.ax_polytope.get_xlim())
        self.current_polytope_ylim = list(self.ax_polytope.get_ylim())

        # Get current data
        row = self.data.kinematics.iloc[self.current_frame]
        time = row['time']
        hip_angle = row['hip_angle']
        knee_angle = row['knee_angle']
        ankle_angle = row['ankle_angle']
        phase = row['phase']

        # Update labels
        progress = 100 * self.current_frame / (self.n_frames - 1) if self.n_frames > 1 else 0
        self.time_label.config(text=f"Time: {time:.3f} s ({progress:.1f}%)")
        self.phase_label.config(text=f"Phase: {phase.replace('_', ' ').title()}")

        # Get selected muscles
        selected_muscles = [m for m, var in self.muscle_vars.items()
                          if var.get()]

        # === UPDATE LEG SKELETON ===
        self.ax_leg.clear()
        self.ax_leg.set_aspect('equal')
        self.ax_leg.set_xlabel('X (m)')
        self.ax_leg.set_ylabel('Y (m)')
        self.ax_leg.set_title('Leg Posture (Sagittal Plane)')
        self.ax_leg.grid(True, alpha=0.3)

        # Compute joint positions
        hip, knee, ankle, toe = self.leg_model.compute_positions(
            hip_angle, knee_angle, ankle_angle
        )

        # Draw ground
        self.ax_leg.plot([-0.8, 0.8], [-1.0, -1.0], 'k-', linewidth=2)
        self.ax_leg.fill_between([-0.8, 0.8], [-1.0, -1.0], [-1.3, -1.3],
                                color='gray', alpha=0.3)

        # Draw skeleton
        # Thigh
        self.ax_leg.plot([hip[0], knee[0]], [hip[1], knee[1]],
                        'o-', color='#2E4057', linewidth=4, markersize=8,
                        label='Thigh')
        # Shank
        self.ax_leg.plot([knee[0], ankle[0]], [knee[1], ankle[1]],
                        'o-', color='#048A81', linewidth=4, markersize=8,
                        label='Shank')
        # Foot
        self.ax_leg.plot([ankle[0], toe[0]], [ankle[1], toe[1]],
                        'o-', color='#54C6EB', linewidth=4, markersize=8,
                        label='Foot')

        # Draw muscles (simplified representation)
        activations = self.data.activations.iloc[self.current_frame]

        for muscle in selected_muscles:
            if muscle in activations:
                activation = activations[muscle]

                # Determine muscle attachment points (simplified)
                if 'glut' in muscle or 'iliacus' in muscle:
                    # Hip muscles
                    start = hip + np.array([-0.05, 0.05])
                    end = (hip + knee) / 2
                elif 'vast' in muscle or 'rectus' in muscle:
                    # Quadriceps
                    start = (hip + knee) / 2
                    end = knee
                elif 'gastroc' in muscle or 'soleus' in muscle:
                    # Calf
                    start = knee
                    end = ankle
                else:
                    continue

                # Draw muscle line
                alpha = 0.3 + 0.7 * activation
                linewidth = 1 + 3 * activation

                self.ax_leg.plot([start[0], end[0]], [start[1], end[1]],
                               color='red', alpha=alpha, linewidth=linewidth)

        # Labels
        self.ax_leg.text(hip[0]-0.1, hip[1]+0.05, 'Hip', fontsize=9)
        self.ax_leg.text(knee[0]-0.1, knee[1]+0.05, 'Knee', fontsize=9)
        self.ax_leg.text(ankle[0]-0.1, ankle[1]+0.05, 'Ankle', fontsize=9)

        # Restore view limits for leg plot
        self.ax_leg.set_xlim(self.current_leg_xlim)
        self.ax_leg.set_ylim(self.current_leg_ylim)

        # === UPDATE FORCE POLYTOPE ===
        self.ax_polytope.clear()
        self.ax_polytope.set_xlabel('Horizontal Force Fx (N)')
        self.ax_polytope.set_ylabel('Vertical Force Fy (N)')
        self.ax_polytope.set_title('Feasible Force Polytope')
        self.ax_polytope.grid(True, alpha=0.3)
        self.ax_polytope.axhline(y=0, color='k', linewidth=0.5)
        self.ax_polytope.axvline(x=0, color='k', linewidth=0.5)

        if len(selected_muscles) > 0:
            # Get moment arms
            moment_arms = self.get_moment_arms(self.current_frame)

            # Get max forces
            muscle_forces = {}
            for muscle in selected_muscles:
                muscle_forces[muscle] = self.data.muscles.loc[muscle, 'F0_max']

            # Compute Jacobian
            jacobian = self.polytope_computer.compute_jacobian(
                hip_angle, knee_angle, ankle_angle,
                self.leg_model.thigh_length,
                self.leg_model.shank_length
            )

            # Compute polytope
            vertices = self.polytope_computer.compute_polytope(
                moment_arms, muscle_forces, jacobian, selected_muscles
            )

            # Plot polytope
            if len(vertices) > 2:
                polygon = Polygon(vertices, alpha=0.3, facecolor='blue',
                                edgecolor='blue', linewidth=2)
                self.ax_polytope.add_patch(polygon)

                # Plot vertices
                self.ax_polytope.plot(vertices[:, 0], vertices[:, 1],
                                    'bo', markersize=4)

            # Frailty threshold
            body_weight = self.data.segment_lengths['subject_mass'] * 9.81
            threshold = body_weight * self.data.config['frailty']['threshold_multiplier']

            self.ax_polytope.axhline(y=threshold, color='red',
                                    linestyle='--', linewidth=2,
                                    label=f'Frailty Threshold ({threshold:.0f} N)')

            self.ax_polytope.legend()

        # Restore view limits for polytope plot
        self.ax_polytope.set_xlim(self.current_polytope_xlim)
        self.ax_polytope.set_ylim(self.current_polytope_ylim)

        # === UPDATE INFO TEXT ===
        self.info_text.delete(1.0, tk.END)
        info = f"""
Frame: {self.current_frame}/{self.n_frames-1}
Time: {time:.3f} s
Phase: {phase}

Joint Angles:
  Hip:   {hip_angle:6.1f}°
  Knee:  {knee_angle:6.1f}°
  Ankle: {ankle_angle:6.1f}°

Muscles: {len(selected_muscles)}/{len(self.muscle_names)}
"""
        self.info_text.insert(1.0, info)

        # Redraw canvas
        self.canvas.draw()


def generate_sample_data():
    """Generate sample data files for testing"""

    data_path = Path("data")

    # Remove existing data directory if it exists
    if data_path.exists():
        import shutil
        shutil.rmtree(data_path)

    # Create fresh directories
    data_path.mkdir(exist_ok=True)
    (data_path / "muscle_data").mkdir(exist_ok=True)
    (data_path / "moment_arms").mkdir(exist_ok=True)
    (data_path / "kinematics").mkdir(exist_ok=True)

    print("Generating sample data files...")

    # Sample muscle parameters
    muscles = [
        'iliacus', 'glut_max', 'glut_med', 'add_long',
        'biceps_fem_lh', 'biceps_fem_sh', 'semitend', 'semimem',
        'rectus_fem', 'vast_lat', 'vast_med', 'vast_int',
        'gastroc_med', 'gastroc_lat', 'soleus',
        'tib_ant', 'tib_post', 'peron_brev'
    ]

    muscle_params = pd.DataFrame({
        'muscle_name': muscles,
        'F0_max': [1000, 1500, 800, 600, 900, 600, 900, 1100,
                   1200, 2000, 1800, 1900, 1600, 1400, 4000,
                   1200, 1400, 500],
        'optimal_fiber_length': [0.10, 0.14, 0.09, 0.11, 0.11, 0.10,
                                0.20, 0.08, 0.08, 0.08, 0.09, 0.09,
                                0.05, 0.06, 0.03, 0.10, 0.03, 0.05],
        'tendon_slack_length': [0.09, 0.13, 0.08, 0.10, 0.34, 0.10,
                               0.26, 0.35, 0.35, 0.16, 0.13, 0.14,
                               0.41, 0.39, 0.27, 0.22, 0.31, 0.16],
        'pennation_angle': [0.13, 0.09, 0.15, 0.10, 0.00, 0.40,
                           0.09, 0.26, 0.09, 0.09, 0.09, 0.06,
                           0.30, 0.15, 0.44, 0.09, 0.21, 0.09],
        'PCSA': [45, 65, 35, 28, 32, 25, 33, 40, 42, 70, 65, 68,
                 58, 52, 140, 42, 48, 18]
    })
    muscle_params.set_index('muscle_name', inplace=True)
    muscle_params.to_csv(data_path / "muscle_data" / "muscle_parameters.csv")
    print(f"✓ Created muscle_parameters.csv")

    # Generate time series (1 second, 150 samples)
    n_samples = 150
    time = np.linspace(0, 1, n_samples)

    # Joint angles (simplified stair climbing pattern)
    hip_angles = -20 + 15 * np.sin(2 * np.pi * time)
    knee_angles = 5 + 60 * (np.sin(2 * np.pi * time - np.pi/4) + 1) / 2
    ankle_angles = -10 + 20 * np.sin(2 * np.pi * time + np.pi/2)

    # Phase labels
    phases = []
    for t in time:
        if t < 0.15:
            phases.append('heel_strike')
        elif t < 0.3:
            phases.append('loading')
        elif t < 0.6:
            phases.append('mid_stance')
        elif t < 0.8:
            phases.append('push_off')
        else:
            phases.append('swing')

    kinematics = pd.DataFrame({
        'time': time,
        'hip_angle': hip_angles,
        'knee_angle': knee_angles,
        'ankle_angle': ankle_angles,
        'phase': phases
    })
    kinematics.to_csv(data_path / "kinematics" / "joint_angles.csv", index=False)
    print(f"✓ Created joint_angles.csv")

    # Segment lengths
    segments = {
        'thigh_length': 0.45,
        'shank_length': 0.43,
        'foot_length': 0.25,
        'subject_mass': 75.0,
        'subject_height': 1.75
    }
    segment_file = data_path / "kinematics" / "segment_lengths.json"
    with open(segment_file, 'w') as f:
        json.dump(segments, f, indent=2)
    # Verify file was written
    with open(segment_file, 'r') as f:
        verify = f.read()
    print(f"✓ Created segment_lengths.json ({len(verify)} bytes)")

    # Generate moment arms (simplified)
    ma_data = []
    for t in time:
        for muscle in muscles:
            # Simplified moment arm patterns
            if 'glut' in muscle:
                hip_ma = -0.05
                knee_ma = 0.0
                ankle_ma = 0.0
            elif 'vast' in muscle or 'rectus' in muscle:
                hip_ma = 0.02 if 'rectus' in muscle else 0.0
                knee_ma = 0.045
                ankle_ma = 0.0
            elif 'gastroc' in muscle:
                hip_ma = 0.0
                knee_ma = -0.02
                ankle_ma = 0.05
            elif 'soleus' in muscle:
                hip_ma = 0.0
                knee_ma = 0.0
                ankle_ma = 0.05
            elif 'iliacus' in muscle:
                hip_ma = 0.065
                knee_ma = 0.0
                ankle_ma = 0.0
            else:
                hip_ma = 0.0
                knee_ma = 0.0
                ankle_ma = 0.0

            ma_data.append({
                'time': t,
                'muscle': muscle,
                'hip_moment_arm': hip_ma,
                'knee_moment_arm': knee_ma,
                'ankle_moment_arm': ankle_ma
            })

    ma_df = pd.DataFrame(ma_data)
    ma_df.to_csv(data_path / "moment_arms" / "moment_arm_matrix.csv", index=False)
    print(f"✓ Created moment_arm_matrix.csv")

    # Generate activations (simplified patterns)
    activation_data = {'time': time}
    for muscle in muscles:
        if 'glut' in muscle or 'vast' in muscle or 'soleus' in muscle:
            # High during stance
            activation_data[muscle] = 0.2 + 0.6 * (np.sin(2*np.pi*time - np.pi/2) + 1) / 2
        elif 'iliacus' in muscle or 'tib_ant' in muscle:
            # High during swing
            activation_data[muscle] = 0.1 + 0.5 * (np.sin(2*np.pi*time + np.pi/2) + 1) / 2
        else:
            activation_data[muscle] = 0.1 + 0.3 * np.random.rand(n_samples)

    activations = pd.DataFrame(activation_data)
    activations.to_csv(data_path / "muscle_data" / "activation_timeseries.csv",
                      index=False)
    print(f"✓ Created activation_timeseries.csv")

    # Generate forces
    force_data = {'time': time}
    for muscle in muscles:
        f_max = muscle_params.loc[muscle, 'F0_max']
        force_data[muscle] = activation_data[muscle] * f_max

    forces = pd.DataFrame(force_data)
    forces.to_csv(data_path / "muscle_data" / "muscle_forces.csv", index=False)
    print(f"✓ Created muscle_forces.csv")

    # Config file
    config = {
        'visualization': {
            'figure_width': 14,
            'figure_height': 8,
            'animation_fps': 30
        },
        'muscles': {
            'groups': {
                'hip_extensors': ['glut_max', 'biceps_fem_lh', 'semitend', 'semimem'],
                'quadriceps': ['rectus_fem', 'vast_lat', 'vast_med', 'vast_int'],
                'calf': ['gastroc_med', 'gastroc_lat', 'soleus'],
                'hip_flexors': ['iliacus', 'rectus_fem'],
                'stabilizers': ['glut_med', 'tib_ant']
            }
        },
        'frailty': {
            'threshold_multiplier': 1.1
        },
        'colors': {
            'hip_extensors': '#8B0000',
            'quadriceps': '#FFD700',
            'calf': '#4169E1',
            'hip_flexors': '#228B22',
            'stabilizers': '#9932CC'
        }
    }

    with open(data_path / "config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"✓ Created config.yaml")

    print("\n✓ Sample data generated successfully!")


def main():
    """Main entry point"""

    # Check if data exists and is complete
    data_loader = DataLoader("data")
    files_ok, missing = data_loader.check_files_exist()

    if not files_ok:
        print("Missing or empty data files:")
        for f in missing:
            print(f"  - {f}")
        print("\nGenerating sample data...")
        generate_sample_data()
        print("")

    # Load data
    try:
        data_loader.load_all()
    except Exception as e:
        print(f"\n✗ Error loading data: {e}")
        print("\nRegenerating sample data...")
        generate_sample_data()
        data_loader = DataLoader("data")
        data_loader.load_all()

    # Create GUI
    root = tk.Tk()
    app = StairClimbingGUI(root, data_loader)

    # Run
    print("\n" + "="*60)
    print("GUI Started Successfully!")
    print("="*60)
    print("\nControls:")
    print("  - Use slider to change posture")
    print("  - Check/uncheck muscles to see their contribution")
    print("  - Click Play to animate")
    print("  - Adjust speed with dropdown")
    print("  - Use toolbar buttons to pan/zoom plots")
    print("  - Zoom/pan persists during animation")
    print("  - Click 'Export Video' to save animation")
    print("\nClose window to exit.")
    print("="*60 + "\n")

    root.mainloop()


if __name__ == "__main__":
    main()