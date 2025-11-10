# -*- coding: utf-8 -*-
"""
Simulator plotting functions:

plotVehicleStates(simTime, simData, figNo) 
plotControls(simTime, simData, vehicle, figNo)
def plot3D(simData, numDataPoints, FPS, filename, figNo)

Author:     Thor I. Fossen
"""

import math
import matplotlib.pyplot as plt
import numpy as np
from python_vehicle_simulator.lib.gnc import ssa
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation

legendSize = 10  # legend size
figSize1 = [25, 13]  # figure1 size in cm
figSize2 = [25, 13]  # figure2 size in cm
dpiValue = 150  # figure dpi value


def R2D(value):  # radians to degrees
    return value * 180 / math.pi


def cm2inch(value):  # inch to cm
    return value / 2.54

# AUV Geometry 

class AUVGeometry:
    """
    Handles the generation and transformation of the 3D AUV model geometry.
    """
    def __init__(self, geometry):
        self.geometry = geometry
        self.L = geometry['l'] # Store total length for COM shift
        self.generate_base_geometry()

    def generate_base_geometry(self):
        """Generate the AUV geometry"""
        geo = self.geometry
        
        r_max = geo['d'] / 2
        num_x_points = 20 # Reduced for faster animation
        num_theta_points = 18 # Reduced for faster animation
        theta = np.linspace(0, 2 * np.pi, num_theta_points)
        z_offset = 0.001
        
        # NOSE SECTION (Elipsoid)
        x_nose = np.linspace(0, geo['a'], num_x_points)
        r_nose = r_max - (r_max - geo['a_offset']) * (1 - x_nose / geo['a'])**geo['n']
        
        X_nose, THETA_nose = np.meshgrid(x_nose, theta)
        R_nose, _ = np.meshgrid(r_nose, theta)
        Y_nose = R_nose * np.cos(THETA_nose)
        Z_nose = R_nose * np.sin(THETA_nose)

        # MID-SECTION (Cylinder)
        mid_section_length = geo['lf'] - geo['a']
        num_x_mid_points = max(2, int(num_x_points * (mid_section_length / geo['a'])))
        x_mid = np.linspace(geo['a'], geo['lf'], num_x_mid_points)
        r_mid = np.full_like(x_mid, r_max)
        
        X_mid, THETA_mid = np.meshgrid(x_mid, theta)
        R_mid, _ = np.meshgrid(r_mid, theta)
        Y_mid = R_mid * np.cos(THETA_mid)
        Z_mid = R_mid * np.sin(THETA_mid)

        # TAIL SECTION (Power Series Curve of Revolution)
        c = geo['l'] - geo['lf']
        num_x_tail_points = max(2, int(num_x_points * (c / geo['a'])))
        x_tail = np.linspace(geo['lf'], geo['l'], num_x_tail_points)
        
        x_norm_tail = (x_tail - geo['lf']) / c
        r_tail = geo['c_offset'] + (r_max - geo['c_offset']) * (1 - x_norm_tail**geo['n'])
        
        X_tail, THETA_tail = np.meshgrid(x_tail, theta)
        R_tail, _ = np.meshgrid(r_tail, theta)
        Y_tail = R_tail * np.cos(THETA_tail)
        Z_tail = R_tail * np.sin(THETA_tail)
        
        r_final = geo['c_offset']

        # DVL (Doppler Velocity Log) Box
        dvl_len = 0.1
        dvl_width = 0.08
        dvl_height = 0.03
        x_dvl_start = geo['lf'] - dvl_len - 0.05
        x_dvl_end = x_dvl_start + dvl_len
        y_dvl_half = dvl_width / 2
        z_dvl_top = -r_max # Note: Z-down in body frame
        z_dvl_bottom = z_dvl_top - dvl_height
        v = np.array([
            [x_dvl_start, -y_dvl_half, z_dvl_top], [x_dvl_end, -y_dvl_half, z_dvl_top],
            [x_dvl_end, y_dvl_half, z_dvl_top], [x_dvl_start, y_dvl_half, z_dvl_top],
            [x_dvl_start, -y_dvl_half, z_dvl_bottom], [x_dvl_end, -y_dvl_half, z_dvl_bottom],
            [x_dvl_end, y_dvl_half, z_dvl_bottom], [x_dvl_start, y_dvl_half, z_dvl_bottom]
        ])
        dvl_faces = [
            [v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]], [v[0], v[1], v[5], v[4]],
            [v[2], v[3], v[7], v[6]], [v[0], v[3], v[7], v[4]], [v[1], v[2], v[6], v[5]]
        ]

        # Fins
        fin_length = 0.12
        fin_span = 0.1
        fin_taper_ratio = 0.8
        fin_x_end = geo['l'] - 0.025 
        fin_x_start = fin_x_end - fin_length
        
        x_norm_fin_start = (fin_x_start - geo['lf']) / c
        r_fin_start = geo['c_offset'] + (r_max - geo['c_offset']) * (1 - x_norm_fin_start**geo['n'])
        x_norm_fin_end = (fin_x_end - geo['lf']) / c
        r_fin_end = geo['c_offset'] + (r_max - geo['c_offset']) * (1 - x_norm_fin_end**geo['n'])
        
        fin_verts = []
        # Starboard (Y+) Fin
        v1 = [fin_x_start, r_fin_start, 0]; v2 = [fin_x_start, r_fin_start + fin_span, 0]
        v3 = [fin_x_end, r_fin_end + fin_span * fin_taper_ratio, 0]; v4 = [fin_x_end, r_fin_end, 0]
        fin_verts.append([v1, v2, v3, v4])
        # Port (Y-) Fin
        v1 = [fin_x_start, -r_fin_start, 0]; v2 = [fin_x_start, -(r_fin_start + fin_span), 0]
        v3 = [fin_x_end, -(r_fin_end + fin_span * fin_taper_ratio), 0]; v4 = [fin_x_end, -r_fin_end, 0]
        fin_verts.append([v1, v2, v3, v4])
        # Top (Z-) Fin (Z is down, so -r is "up" visually on plot)
        v1 = [fin_x_start, 0, -r_fin_start]; v2 = [fin_x_start, 0, -(r_fin_start + fin_span)]
        v3 = [fin_x_end, 0, -(r_fin_end + fin_span * fin_taper_ratio)]; v4 = [fin_x_end, 0, -r_fin_end]
        fin_verts.append([v1, v2, v3, v4])
        # Bottom (Z+) Fin
        v1 = [fin_x_start, 0, r_fin_start]; v2 = [fin_x_start, 0, r_fin_start + fin_span]
        v3 = [fin_x_end, 0, r_fin_end + fin_span * fin_taper_ratio]; v4 = [fin_x_end, 0, r_fin_end]
        fin_verts.append([v1, v2, v3, v4])
        
        # Store all geometry
        self.base_geometry = {
            'nose': (X_nose, Y_nose, Z_nose),
            'mid': (X_mid, Y_mid, Z_mid),
            'tail': (X_tail, Y_tail, Z_tail),
            'dvl_faces': dvl_faces,
        }
        self.fins = fin_verts

    def rotation_matrix(self, roll, pitch, yaw):
        """Create rotation matrix from Euler angles (ZYX convention)"""
        # Rotation around x-axis (roll)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        # Rotation around y-axis (pitch)
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        # Rotation around z-axis (yaw)
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        # Combined rotation: Rz * Ry * Rx
        return Rz @ Ry @ Rx
    
    def transform_geometry(self, X, Y, Z, position, orientation):
        """Apply position and orientation to geometry points"""
        # Apply COM shift: subtract L/2 from X before rotation
        points = np.stack([X.flatten() - (self.L / 2.0), Y.flatten(), Z.flatten()])
        
        R = self.rotation_matrix(*orientation)
        rotated = R @ points
        translated = rotated + position.reshape(3, 1)

        X_new = translated[0].reshape(X.shape)
        Y_new = translated[1].reshape(Y.shape)
        Z_new = translated[2].reshape(Z.shape)

        return X_new, Y_new, Z_new
    
    def transform_fins(self, position, orientation):
        """Transform fin vertices"""
        transformed_fins = []
        R = self.rotation_matrix(*orientation)
        for fin in self.fins:
            fin_array = np.array(fin)
            # Apply COM shift: subtract L/2 from x-coordinates
            fin_array[:, 0] -= (self.L / 2.0)
            
            rotated = (R @ fin_array.T).T
            translated = rotated + position
            transformed_fins.append(translated)
        return transformed_fins
    
    def transform_dvl(self, position, orientation):
        """Transform DVL box"""
        R = self.rotation_matrix(*orientation)
        transformed_faces = []
        for face in self.base_geometry['dvl_faces']:
            face_array = np.array(face)
            
            # Apply COM shift: subtract L/2 from x-coordinates
            face_array_shifted = face_array.copy()
            face_array_shifted[:, 0] -= (self.L / 2.0)
            
            rotated = (R @ face_array_shifted.T).T
            translated = rotated + position
            transformed_faces.append(translated)
        return transformed_faces

# plotVehicleStates(simTime, simData, figNo) plots the 6-DOF vehicle
# position/attitude and velocities versus time in figure no. figNo
def plotVehicleStates(simTime, simData, figNo):

    # Time vector
    t = simTime

    # State vectors
    x = simData[:, 0]
    y = simData[:, 1]
    z = simData[:, 2]
    phi = R2D(ssa(simData[:, 3]))
    theta = R2D(ssa(simData[:, 4]))
    psi = R2D(ssa(simData[:, 5]))
    u = simData[:, 6]
    v = simData[:, 7]
    w = simData[:, 8]
    p = R2D(simData[:, 9])
    q = R2D(simData[:, 10])
    r = R2D(simData[:, 11])

    # Speed
    U = np.sqrt(np.multiply(u, u) + np.multiply(v, v) + np.multiply(w, w))

    beta_c  = R2D(ssa(np.arctan2(v,u)))   # crab angle, beta_c    
    alpha_c = R2D(ssa(np.arctan2(w,u)))   # flight path angle
    chi = R2D(ssa(simData[:, 5] + np.arctan2(v, u)))  # course angle, chi=psi+beta_c

    # Plots
    plt.figure(
        figNo, figsize=(cm2inch(figSize1[0]), cm2inch(figSize1[1])), dpi=dpiValue
    )
    plt.grid()

    plt.subplot(3, 3, 1)
    plt.plot(y, x, label='Trajectory')
    plt.plot(y[0], x[0], 'go', markersize=5, label='Start') # Start marker
    plt.plot(y[-1], x[-1], 'rs', markersize=5, label='End') # End marker
    plt.legend(fontsize=legendSize)
    plt.xlabel('North (m)')
    plt.ylabel('East (m)')
    plt.title('North-East Trajectory')
    plt.grid()

    plt.subplot(3, 3, 2)
    plt.plot(t, z)
    plt.legend(["Depth (m)"], fontsize=legendSize)
    plt.grid()

    plt.title("Vehicle states", fontsize=12)

    plt.subplot(3, 3, 3)
    plt.plot(t, phi, t, theta)
    plt.legend(["Roll angle (deg)", "Pitch angle (deg)"], fontsize=legendSize)
    plt.grid()

    plt.subplot(3, 3, 4)
    plt.plot(t, U)
    plt.legend(["Speed (m/s)"], fontsize=legendSize)
    plt.grid()

    plt.subplot(3, 3, 5)
    plt.plot(t, chi)
    plt.legend(["Course angle (deg)"], fontsize=legendSize)
    plt.grid()

    plt.subplot(3, 3, 6)
    plt.plot(t, theta, t, alpha_c)
    plt.legend(["Pitch angle (deg)", "Flight path angle (deg)"], fontsize=legendSize)
    plt.grid()

    plt.subplot(3, 3, 7)
    plt.plot(t, u, t, v, t, w)
    plt.xlabel("Time (s)", fontsize=12)
    plt.legend(
        ["Surge velocity (m/s)", "Sway velocity (m/s)", "Heave velocity (m/s)"],
        fontsize=legendSize,
    )
    plt.grid()

    plt.subplot(3, 3, 8)
    plt.plot(t, p, t, q, t, r)
    plt.xlabel("Time (s)", fontsize=12)
    plt.legend(
        ["Roll rate (deg/s)", "Pitch rate (deg/s)", "Yaw rate (deg/s)"],
        fontsize=legendSize,
    )
    plt.grid()

    plt.subplot(3, 3, 9)
    plt.plot(t, psi, t, beta_c)
    plt.xlabel("Time (s)", fontsize=12)
    plt.legend(["Yaw angle (deg)", "Crab angle (deg)"], fontsize=legendSize)
    plt.grid()


# plotControls(simTime, simData) plots the vehicle control inputs versus time
# in figure no. figNo
def plotControls(simTime, simData, vehicle, figNo):

    DOF = 6

    # Time vector
    t = simTime

    plt.figure(
        figNo, figsize=(cm2inch(figSize2[0]), cm2inch(figSize2[1])), dpi=dpiValue
    )

    # Columns and rows needed to plot vehicle.dimU control inputs
    col = 2
    row = int(math.ceil(vehicle.dimU / col))

    # Plot the vehicle.dimU active control inputs
    for i in range(0, vehicle.dimU):

        u_control = simData[:, 2 * DOF + i]  # control input, commands
        u_actual = simData[:, 2 * DOF + vehicle.dimU + i]  # actual control input

        if vehicle.controls[i].find("deg") != -1:  # convert angles to deg
            u_control = R2D(u_control)
            u_actual = R2D(u_actual)

        plt.subplot(row, col, i + 1)
        plt.plot(t, u_control, t, u_actual)
        plt.legend(
            [vehicle.controls[i] + ", command", vehicle.controls[i] + ", actual"],
            fontsize=legendSize,
        )
        plt.xlabel("Time (s)", fontsize=12)
        plt.grid()


# plot3D(simData,numDataPoints,FPS,filename,figNo) plots the vehicles position
# (x, y, z) in 3D in figure no. figNo
def plot3D(simData, numDataPoints, FPS, filename, figNo):
    """
    Animates the AUV's 3D trajectory using the full vehicle geometry,
    with the COM on the trajectory and the nose pointing along the
    physical vehicle heading (psi).
    """        

    # AUV geometry model 
    # Parameters from remus100.py
    L = 1.6 
    D = 0.19
    
    # Parameters from auv_model.py
    old_L = 1.33
    old_a = 0.191
    old_a_offset = 0.0165
    old_c_offset = 0.0368
    old_lf = 0.828
    
    scale_ratio = L / old_L

    auv_geo = {
        'a': old_a * scale_ratio,
        'a_offset': old_a_offset * scale_ratio,
        'c_offset': old_c_offset * scale_ratio,
        'n': 2,
        'd': D,
        'lf': old_lf * scale_ratio,
        'l': L,
    }

    auv_model = AUVGeometry(auv_geo)

    # Downsample simData to numDataPoints
    step = max(1, len(simData // numDataPoints))
    data = simData[::step]

    # Extract states (N, E, D, phi, theta, psi) 
    n = data[:, 0] # North
    e = data[:, 1] # East
    d = data[:, 2] # Down
    phi = data[:, 3] # Roll
    theta = data[:, 4] # Pitch
    psi = data[:, 5] # Yaw

    # Convert to plot coordinates (E, N, U)
    x_plot = e
    y_plot = n
    z_plot = -d #Up is negative down
    

    # # State vectors
    # x = simData[:,0]
    # y = simData[:,1]
    # z = simData[:,2]
    
    # # down-sampling the xyz data points
    # N = y[::len(x) // numDataPoints];
    # E = x[::len(x) // numDataPoints];
    # D = z[::len(x) // numDataPoints];
    
    # # Animation function
    # def anim_function(num, dataSet, line):
        
    #     line.set_data(dataSet[0:2, :num])    
    #     line.set_3d_properties(dataSet[2, :num])    
    #     ax.view_init(elev=10.0, azim=-120.0)
        
    #     return line
    
    # dataSet = np.array([E, N, -D])      # (East, North, Up)
    
    # # Attaching 3D axis to the figure
    # fig = plt.figure(figNo,figsize=(cm2inch(figSize1[0]),cm2inch(figSize1[1])),
    #            dpi=dpiValue)
    # ax = p3.Axes3D(fig, auto_add_to_figure=False)
    # fig.add_axes(ax) 
    
    # # Line/trajectory plot
    # line = plt.plot(dataSet[0], dataSet[1], dataSet[2], lw=2, c='b', label='Trajectory')[0] 

    # # Add Start and End markers
    # ax.plot(dataSet[0, 0], dataSet[1, 0], dataSet[2, 0], 'go', markerfacecolor='green', markersize=5, label='Start')
    # ax.plot(dataSet[0, -1], dataSet[1, -1], dataSet[2, -1], 'rs', markerfacecolor='red', markersize=5, label='End')
    # ax.legend()

    # # Setting the axes properties
    # ax.set_xlabel('X / East')
    # ax.set_ylabel('Y / North')
    # ax.set_zlim3d([-100, 20])                   # default depth = -100 m
    
    # if np.amax(z) > 100.0:
    #     ax.set_zlim3d([-np.amax(z), 20])
        
    # ax.set_zlabel('-Z / Down')

    # # Plot 2D surface for z = 0
    # [x_min, x_max] = ax.get_xlim()
    # [y_min, y_max] = ax.get_ylim()
    # x_grid = np.arange(x_min-20, x_max+20)
    # y_grid = np.arange(y_min-20, y_max+20)
    # [xx, yy] = np.meshgrid(x_grid, y_grid)
    # zz = 0 * xx
    # ax.plot_surface(xx, yy, zz, alpha=0.3)
                    
    # # Title of plot
    # ax.set_title('North-East-Down')
    
    # # Create the animation object
    # ani = animation.FuncAnimation(fig, 
    #                      anim_function, 
    #                      frames=numDataPoints, 
    #                      fargs=(dataSet,line),
    #                      interval=200, 
    #                      blit=False,
    #                      repeat=True)
    
    # # Save the 3D animation as a gif file
    # ani.save(filename, writer=animation.PillowWriter(fps=FPS))