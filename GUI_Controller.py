import customtkinter as ctk
import numpy as np
from fwk_solver import forward_kinematics, rotation_matrix_to_euler_angles, joint_limits_deg
from ik_solver import inverse_kinematics
from coppeliasim_interface import CoppeliaSimInterface

class RobotControlUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Robot Arm Control - Yaskawa GP8")
        self.root.geometry("750x480")
        self.root.resizable(False, False)
        ctk.set_appearance_mode("light")
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.sim_interface = CoppeliaSimInterface()
        self.fwk_sliders = []
        self.fwk_entries = []
        self.status_label = None
        self.moving_indicator = None
        self.is_moving = False
        self.saved_pose = None
        self.gripper_state = False
        self.wrist_flipped = False

        self.ik_pos = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.ik_orient = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        self.pos_step = 0.01
        self.orient_step = 1.0
        self.ik_entries = {}
        self.status_message = None

        self.create_ui()
        self.root.after(100, self.update_ui)

    def create_ui(self):
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=(5, 5))

        # FWK Panel (Left)
        fwk_frame = ctk.CTkFrame(main_frame)
        fwk_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        ctk.CTkLabel(fwk_frame, text="Forward Kinematics", font=("Arial", 16, "bold")).pack(pady=(5, 5))

        step_frame = ctk.CTkFrame(fwk_frame, fg_color="transparent")
        step_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(step_frame, text="Step size (°):", width=80).pack(side="left", padx=10)
        self.joint_step_entry = ctk.CTkEntry(step_frame, width=50)
        self.joint_step_entry.insert(0, "1.0")
        self.joint_step_entry.pack(side="left", padx=5)

        for i in range(6):
            limits = joint_limits_deg[i]
            joint_frame = ctk.CTkFrame(fwk_frame, fg_color="transparent")
            joint_frame.pack(fill="x", pady=1)

            ctk.CTkLabel(joint_frame, text=f"Joint {i+1}:", width=60).pack(side="left", padx=5)
            slider = ctk.CTkSlider(joint_frame, from_=limits['min'], to=limits['max'], number_of_steps=340, width=180)
            slider.set(0)
            slider.pack(side="left", padx=5)
            self.fwk_sliders.append(slider)
            entry = ctk.CTkEntry(joint_frame, width=50)
            entry.insert(0, "0.0")
            entry.pack(side="left", padx=5)
            self.fwk_entries.append(entry)
            slider.configure(command=lambda val, e=entry, s=slider: self.update_entry_from_slider(e, s))
            entry.bind("<Return>", lambda e, s=slider, ent=entry: self.update_slider_from_entry(s, ent))

        pos_frame = ctk.CTkFrame(fwk_frame, fg_color="transparent")
        pos_frame.pack(fill="x", pady=5, anchor="w")
        ctk.CTkLabel(pos_frame, text="Actual End-effector position", font=("Arial", 12, "bold")).pack(padx=(35, 0), anchor="w")
        self.fwk_pos_labels = {}
        self.fwk_orient_labels = {}
        for label, key in [("X:", "x"), ("Y:", "y"), ("Z:", "z")]:
            subframe = ctk.CTkFrame(pos_frame, fg_color="transparent")
            subframe.pack(fill="x", pady=1)
            ctk.CTkLabel(subframe, text=label, width=30).pack(side="left", padx=5)
            value_label = ctk.CTkLabel(subframe, text="0.0", width=50)
            value_label.pack(side="left", padx=5)
            self.fwk_pos_labels[key] = value_label
            orient_label = f"R{key.upper()}:"
            ctk.CTkLabel(subframe, text=orient_label, width=30).pack(side="left", padx=20)
            orient_value = ctk.CTkLabel(subframe, text="0.0", width=50)
            orient_value.pack(side="left", padx=5)
            self.fwk_orient_labels[key] = orient_value

        # IK Panel (Right)
        ik_frame = ctk.CTkFrame(main_frame)
        ik_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        ctk.CTkLabel(ik_frame, text="Inverse Kinematics", font=("Arial", 16, "bold")).grid(row=0, column=0, columnspan=3, pady=(5, 5))

        step_frame = ctk.CTkFrame(ik_frame, fg_color="transparent")
        step_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=2)
        ctk.CTkLabel(step_frame, text="Pos Step (m):", width=80).pack(side="left", padx=5)
        self.pos_step_entry = ctk.CTkEntry(step_frame, width=50)
        self.pos_step_entry.insert(0, "0.01")
        self.pos_step_entry.pack(side="left", padx=5)
        ctk.CTkLabel(step_frame, text="Ori Step (°):", width=80).pack(side="left", padx=(35,0))
        self.orient_step_entry = ctk.CTkEntry(step_frame, width=50)
        self.orient_step_entry.insert(0, "1.0")
        self.orient_step_entry.pack(side="left", padx=5)

        # X, Y Control
        xy_frame = ctk.CTkFrame(ik_frame, fg_color="transparent")
        xy_frame.grid(row=2, column=0, padx=(5, 0), pady=5)  
        ctk.CTkLabel(xy_frame, text="X, Y Control", font=("Arial", 12, "bold")).pack()

        xy_subframe = ctk.CTkFrame(xy_frame, fg_color="transparent") 
        xy_subframe.pack(pady=2)
        self.xy_up_btn = ctk.CTkButton(xy_subframe, text="▲", width=40, height=40, command=lambda: self.adjust_ik_pos("y", 1))
        self.xy_up_btn.grid(row=0, column=1, padx=5, pady=2)
        self.xy_left_btn = ctk.CTkButton(xy_subframe, text="◄", width=40, height=40, command=lambda: self.adjust_ik_pos("x", -1))
        self.xy_left_btn.grid(row=1, column=0, padx=5, pady=2)
        self.xy_home_btn = ctk.CTkButton(xy_subframe, text="H", width=40, height=40, command=self.go_home) 
        self.xy_home_btn.grid(row=1, column=1, padx=5, pady=2)
        self.xy_right_btn = ctk.CTkButton(xy_subframe, text="►", width=40, height=40, command=lambda: self.adjust_ik_pos("x", 1))
        self.xy_right_btn.grid(row=1, column=2, padx=5, pady=2)
        self.xy_down_btn = ctk.CTkButton(xy_subframe, text="▼", width=40, height=40, command=lambda: self.adjust_ik_pos("y", -1))
        self.xy_down_btn.grid(row=2, column=1, padx=5, pady=2)

        # Orientation Control 
        orient_frame = ctk.CTkFrame(ik_frame, fg_color="transparent")
        orient_frame.grid(row=2, column=1, padx=(50, 5), pady=5, sticky="w") 
        ctk.CTkLabel(orient_frame, text="Orientation Control", font=("Arial", 12, "bold")).pack(padx=(30,0))
        for label, key in [("Roll:", "roll"), ("Pitch:", "pitch"), ("Yaw:", "yaw")]:
            subframe = ctk.CTkFrame(orient_frame, fg_color="transparent") 
            subframe.pack(fill="x", pady=2)
            ctk.CTkLabel(subframe, text=label, width=60).pack(side="left", padx=5)
            dec_btn = ctk.CTkButton(subframe, text="◄", width=30, command=lambda k=key: self.adjust_ik_orient(k, -1))
            dec_btn.pack(side="left", padx=2)
            inc_btn = ctk.CTkButton(subframe, text="►", width=30, command=lambda k=key: self.adjust_ik_orient(k, 1))
            inc_btn.pack(side="left", padx=2)

        # Z Control, Gripper, and IK Input Fields
        z_gripper_input_frame = ctk.CTkFrame(ik_frame, fg_color="transparent") 
        z_gripper_input_frame.grid(row=3, column=0, columnspan=3, pady=5, sticky="ew")

        # Z Control
        z_frame = ctk.CTkFrame(z_gripper_input_frame, fg_color="transparent")  
        z_frame.pack(side="left", padx=5)
        ctk.CTkLabel(z_frame, text="Z Control", font=("Arial", 12, "bold")).pack()
        z_subframe = ctk.CTkFrame(z_frame, fg_color="transparent")
        z_subframe.pack(pady=2)
        self.z_up_btn = ctk.CTkButton(z_subframe, text="▲", width=40, height=40, command=lambda: self.adjust_ik_pos("z", 1))
        self.z_up_btn.pack(side="left", padx=5)
        self.z_down_btn = ctk.CTkButton(z_subframe, text="▼", width=40, height=40, command=lambda: self.adjust_ik_pos("z", -1))
        self.z_down_btn.pack(side="left", padx=5)

        # Gripper Control
        gripper_frame = ctk.CTkFrame(z_gripper_input_frame, fg_color="transparent") 
        gripper_frame.pack(side="left", padx=5)
        ctk.CTkLabel(gripper_frame, text="Gripper", font=("Arial", 12, "bold")).pack()
        self.gripper_btn = ctk.CTkButton(gripper_frame, text="Close" if self.gripper_state else "Open", width=60, command=self.toggle_gripper)
        self.gripper_btn.pack(pady=2)

        # IK Position and Orientation Input Fields
        ik_pos_frame = ctk.CTkFrame(z_gripper_input_frame, fg_color="transparent") 
        ik_pos_frame.pack(side="left", padx=5)
        self.ik_pos_labels = {}
        self.ik_orient_labels = {}
        for idx, (label, key) in enumerate([("X:", "x"), ("Y:", "y"), ("Z:", "z")]):
            ctk.CTkLabel(ik_pos_frame, text=label, width=30).grid(row=idx, column=0, padx=5, pady=1)
            entry = ctk.CTkEntry(ik_pos_frame, width=50)
            entry.insert(0, "0.0")
            entry.grid(row=idx, column=1, padx=5, pady=1)
            entry.bind("<Return>", lambda e, k=key: self.update_ik_from_entry(k))
            self.ik_entries[key] = entry
            self.ik_pos_labels[key] = entry 
            orient_label = f"R{key.upper()}:"
            ctk.CTkLabel(ik_pos_frame, text=orient_label, width=30).grid(row=idx, column=2, padx=5, pady=1)
            orient_entry = ctk.CTkEntry(ik_pos_frame, width=50)
            orient_entry.insert(0, "0.0")
            orient_entry.grid(row=idx, column=3, padx=5, pady=1)
            orient_entry.bind("<Return>", lambda e, k=f"r{key}": self.update_ik_from_entry(k))
            self.ik_entries[f"r{key}"] = orient_entry
            self.ik_orient_labels[key] = orient_entry

        # Buttons
        button_frame = ctk.CTkFrame(ik_frame, fg_color="transparent")
        button_frame.grid(row=4, column=0, columnspan=3, pady=5)
        self.flip_wrist_btn = ctk.CTkButton(button_frame, text="Flip Wrist", width=55, command=self.flip_wrist)  
        self.flip_wrist_btn.pack(side="left", padx=5)
        compute_btn = ctk.CTkButton(button_frame, text="Compute IK", width=70, command=self.compute_ik)  
        compute_btn.pack(side="left", padx=5)
        save_btn = ctk.CTkButton(button_frame, text="Save Pose", width=70, command=self.save_pose)  
        save_btn.pack(side="left", padx=5)
        load_btn = ctk.CTkButton(button_frame, text="Load Pose", width=70, command=self.load_pose)  
        load_btn.pack(side="left", padx=5)

        # Status message
        status_frame = ctk.CTkFrame(ik_frame, fg_color="transparent")
        status_frame.grid(row=5, column=0, columnspan=3, pady=2)
        self.status_message = ctk.CTkLabel(status_frame, text="", font=("Arial", 12))
        self.status_message.pack(padx=5)

        # Bottom panel
        bottom_frame = ctk.CTkFrame(self.root)
        bottom_frame.pack(fill="x", padx=10, pady=5)
        
        status_container = ctk.CTkFrame(bottom_frame, fg_color="transparent")
        status_container.pack(side="left", fill="x", expand=True, padx=5)
        self.status_label = ctk.CTkLabel(status_container, text="Connection Status: Disconnected", text_color="red", font=("Arial", 12))
        self.status_label.pack(side="left", padx=5)
        self.moving_indicator = ctk.CTkLabel(status_container, text="●", text_color="gray", font=("Arial", 12))
        self.moving_indicator.pack(side="left", padx=5)
        
        button_container = ctk.CTkFrame(bottom_frame, fg_color="transparent")
        button_container.pack(side="right", padx=5)
        ctk.CTkButton(button_container, text="Connect", command=self.connect_sim).pack(side="left", padx=5)
        ctk.CTkButton(button_container, text="Disconnect", command=self.disconnect_sim).pack(side="left", padx=5)

        main_frame.grid_columnconfigure((0, 1), weight=1)

    def update_entry_from_slider(self, entry, slider):
        entry.delete(0, "end")
        entry.insert(0, f"{slider.get():.1f}")
        self.update_fwk()

    def update_slider_from_entry(self, slider, entry):
        try:
            value = float(entry.get())
            limits = joint_limits_deg[self.fwk_sliders.index(slider)]
            if limits['min'] <= value <= limits['max']:
                slider.set(value)
                self.update_fwk()
        except ValueError:
            pass

    def sync_ik_with_current_pose(self):
        if self.sim_interface.is_connected():
            pos, orient = self.sim_interface.get_end_effector_pose()
            if pos and orient:
                self.ik_pos = {"x": pos[0], "y": pos[1], "z": pos[2]}
                self.ik_orient = {"roll": orient[0], "pitch": orient[1], "yaw": orient[2]}
                self.update_ik_display()

    def adjust_ik_pos(self, axis, direction):
        try:
            self.pos_step = float(self.pos_step_entry.get())
        except ValueError:
            self.pos_step = 0.01
        self.ik_pos[axis] += direction * self.pos_step
        self.update_ik_display()
        
        current_angles = None
        if self.sim_interface.is_connected():
            current_angles = self.sim_interface.get_joint_angles()
        
        self.compute_ik(current_angles)

    def adjust_ik_orient(self, axis, direction):
        try:
            self.orient_step = float(self.orient_step_entry.get())
        except ValueError:
            self.orient_step = 1.0
        self.ik_orient[axis] += direction * self.orient_step
        self.update_ik_display()
        
        current_angles = None
        if self.sim_interface.is_connected():
            current_angles = self.sim_interface.get_joint_angles()
            
        self.compute_ik(current_angles)

    def update_ik_from_entry(self, key):
        try:
            if key in ["x", "y", "z"]:
                self.ik_pos[key] = float(self.ik_entries[key].get())
            else:
                orient_key = key[1]
                self.ik_orient[{"x": "roll", "y": "pitch"}[orient_key]] = float(self.ik_entries[key].get())
            self.update_ik_display()
        except ValueError:
            pass

    def update_ik_display(self):
        for key in ["x", "y", "z"]:
            self.ik_entries[key].delete(0, "end")
            self.ik_entries[key].insert(0, f"{self.ik_pos[key]:.2f}")
        for key, orient_key in [("x", "roll"), ("y", "pitch"), ("z", "yaw")]:
            self.ik_entries[f"r{key}"].delete(0, "end")
            self.ik_entries[f"r{key}"].insert(0, f"{self.ik_orient[orient_key]:.1f}")

    def toggle_gripper(self):
        current_state = self.sim_interface.get_gripper_status()
        new_state = not current_state
        self.sim_interface.set_gripper_status(new_state)
        self.gripper_state = new_state
        self.gripper_btn.configure(text="Open" if not new_state else "Close")
        self.show_status_message(f"Gripper {'opened' if new_state else 'closed'}", "info")

    def flip_wrist(self):
        self.wrist_flipped = not self.wrist_flipped
        self.compute_ik()

    def update_fwk(self):
        angles = [slider.get() for slider in self.fwk_sliders]
        final_T, _ = forward_kinematics(angles + [0])
        pos = final_T[:3, 3]
        euler = [np.degrees(a) for a in rotation_matrix_to_euler_angles(final_T[:3, :3])]
        self.ik_pos = {"x": pos[0], "y": pos[1], "z": pos[2]}
        self.ik_orient = {"roll": euler[0], "pitch": euler[1], "yaw": euler[2]}
        self.update_ik_display()
        self.is_moving = True
        if self.sim_interface.is_connected():
            success = self.sim_interface.set_joint_angles(angles)
            if not success:
                self.show_status_message("Failed to set joint angles", "error")

    def compute_ik(self, current_angles=None):
        if current_angles is None and self.sim_interface.is_connected():
            current_angles = self.sim_interface.get_joint_angles()
            
        wrist_config = "down" if self.wrist_flipped else "up"
        angles = inverse_kinematics(
            self.ik_pos["x"], self.ik_pos["y"], self.ik_pos["z"],
            self.ik_orient["roll"], self.ik_orient["pitch"], self.ik_orient["yaw"],
            wrist_config, current_angles  
        )
        
        if angles:
            for i, (slider, entry) in enumerate(zip(self.fwk_sliders, self.fwk_entries)):
                slider.set(angles[i])
                entry.delete(0, "end")
                entry.insert(0, f"{angles[i]:.1f}")
            if self.sim_interface.is_connected():
                success = self.sim_interface.set_joint_angles(angles)
                if not success:
                    self.show_status_message("Failed to set joint angles", "error")
        else:
            self.show_status_message("Position is Out of Reach", "warning")

    def save_pose(self):
        if self.sim_interface.is_connected():
            pos, orient = self.sim_interface.get_end_effector_pose()
            if pos and orient:
                self.saved_pose = {"x": pos[0], "y": pos[1], "z": pos[2], "roll": orient[0], "pitch": orient[1], "yaw": orient[2]}
                self.show_status_message("Pose saved!", "success")

    def load_pose(self):
        if self.saved_pose:
            self.ik_pos = {"x": self.saved_pose["x"], "y": self.saved_pose["y"], "z": self.saved_pose["z"]}
            self.ik_orient = {"roll": self.saved_pose["roll"], "pitch": self.saved_pose["pitch"], "yaw": self.saved_pose["yaw"]}
            self.update_ik_display()
            self.show_status_message("Pose loaded!", "success")

    def show_status_message(self, message, msg_type="info"):
        if msg_type == "warning":
            self.status_message.configure(text=message, text_color="orange")
        elif msg_type == "error":
            self.status_message.configure(text=message, text_color="red")
        elif msg_type == "success":
            self.status_message.configure(text=message, text_color="green")
        else:
            self.status_message.configure(text=message, text_color=("gray10", "gray90"))
        self.root.after(2000, lambda: self.status_message.configure(text=""))

    def update_ui(self):
        if self.sim_interface.is_connected():
            pos, orient = self.sim_interface.get_end_effector_pose()
            if pos and orient:
                for key in ["x", "y", "z"]:
                    self.fwk_pos_labels[key].configure(text=f"{pos[key == 'x' and 0 or key == 'y' and 1 or 2]:.2f}")
                for key, idx in [("x", 0), ("y", 1), ("z", 2)]:
                    self.fwk_orient_labels[key].configure(text=f"{orient[idx]:.1f}")
            self.status_label.configure(text="Connection Status: Connected", text_color="green")
            self.moving_indicator.configure(text_color="yellow" if self.is_moving else "gray")
            self.is_moving = False
        else:
            self.status_label.configure(text="Connection Status: Disconnected", text_color="red")
            self.moving_indicator.configure(text_color="gray")
            for key in ["x", "y", "z"]:
                self.fwk_pos_labels[key].configure(text="0.0")
                self.fwk_orient_labels[key].configure(text="0.0")
        self.root.after(100, self.update_ui)

    def connect_sim(self):
        if self.sim_interface.connect():
            self.status_label.configure(text="Connection Status: Connected", text_color="green")
            
            angles = self.sim_interface.get_joint_angles()
            if angles:
                for i, (slider, entry) in enumerate(zip(self.fwk_sliders, self.fwk_entries)):
                    slider.set(angles[i])
                    entry.delete(0, "end")
                    entry.insert(0, f"{angles[i]:.1f}")
                
                final_T, _ = forward_kinematics(angles + [0])
                pos = final_T[:3, 3]
                euler = [np.degrees(a) for a in rotation_matrix_to_euler_angles(final_T[:3, :3])]
                self.ik_pos = {"x": pos[0], "y": pos[1], "z": pos[2]}
                self.ik_orient = {"roll": euler[0], "pitch": euler[1], "yaw": euler[2]}
                self.update_ik_display()
                gripper_status = self.sim_interface.get_gripper_status()
                if gripper_status is not None:
                    self.gripper_state = 1
                    self.sim_interface.set_gripper_status(1)
                    self.gripper_btn.configure(text="close" if self.gripper_state else "open")

    def disconnect_sim(self):
        if self.sim_interface.disconnect():
            self.status_label.configure(text="Connection Status: Disconnected", text_color="red")
            for slider, entry in zip(self.fwk_sliders, self.fwk_entries):
                slider.set(0)
                entry.delete(0, "end")
                entry.insert(0, "0.0")
            self.ik_pos = {"x": 0.0, "y": 0.0, "z": 0.0}
            self.ik_orient = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
            self.update_ik_display()

    def go_home(self):
        home_angles = [0] * 6
        if self.sim_interface.is_connected():
            success = self.sim_interface.set_joint_angles(home_angles)
            if not success:
                self.show_status_message("Failed to set joint angles", "error")
        
        for slider, entry in zip(self.fwk_sliders, self.fwk_entries):
            slider.set(0)
            entry.delete(0, "end")
            entry.insert(0, "0.0")
        
        final_T, _ = forward_kinematics(home_angles + [0])
        pos = final_T[:3, 3]
        euler = [np.degrees(a) for a in rotation_matrix_to_euler_angles(final_T[:3, :3])]
        self.ik_pos = {"x": pos[0], "y": pos[1], "z": pos[2]}
        self.ik_orient = {"roll": euler[0], "pitch": euler[1], "yaw": euler[2]}
        
        self.update_ik_display()
        self.show_status_message("Robot moved to home position", "success")

    def on_closing(self):
        if self.sim_interface.is_connected():
            self.sim_interface.disconnect()
        self.root.destroy()

    def run(self):
        self.root.mainloop()    
        
        
if __name__ == "__main__":
    app = RobotControlUI()
    app.run()