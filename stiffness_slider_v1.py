import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from roboticstoolbox import DHRobot, RevoluteDH
from mpl_toolkits.mplot3d import Axes3D

#TODO: Create K = f(torque)
#TODO: Create K = f(muscle activation) - create this matrix depending on dependence of each joint on different muscles. 

def pseudo_inv(J, A):
    A_inv = np.linalg.inv(A)
    B = J @ A_inv @ J.T
    return A_inv @ J.T @ np.linalg.inv(B)

def compute_and_plot(q_deg, Ks_rot, Ks_aa, Ks_ax, Ke):
    q = np.deg2rad(q_deg)
    J = robot.jacob0(q)
    Kj = np.diag([
        Ks_rot, Ks_aa, Ks_ax, Ke, 
        10000, 10000, 10000, 100000  # Fixed high wrist stiffness, the last joint is a dummy for the transform to the thumb
    ])
    J_plus = pseudo_inv(J[:3, :], Kj)
    Ke_matrix = J_plus.T @ Kj @ J_plus
    eigvals, eigvects = np.linalg.eigh(Ke_matrix)
    
    # axes_lengths = 1/np.sqrt(eigvals)

    axes_lengths = eigvals #is sqrt needed?

    # print("Axes lengths: ", axes_lengths)

    ellipsoid = eigvects @ np.diag(axes_lengths) @ sphere #is this correct? do we need to normalise the eigvect
    x_e, y_e, z_e = ellipsoid.reshape(3, *x.shape)

    ellip_plot[0].remove()
    ellip_plot[0] = ax.plot_surface(x_e, y_e, z_e, color='skyblue', alpha=0.7)

    # ax.set_xlim([-50000, 50000])
    # ax.set_ylim([-50000, 50000])
    # ax.set_zlim([-50000, 50000])

    ax.set_box_aspect([1,1,1])
    fig.canvas.draw_idle()

l0_y_Humerus = 0.30 #length of humerus
l0_z_Humerus = 0.03
l0_x_Humerus = 0.02 
l0_z_Ulna = 0.02 
l0_y_Ulna = 0.02 
l0_y_Radius = 0.26 #length of radius
l0_y1_Radius = 0.02
l0_a_Radius = 0.02
l0_d_Radius = 0.02

# In the final frame, the x axis points in the direction of the thumb, and the z axis indicates opposite direction of the palm
robot = DHRobot([
    RevoluteDH(a=0.0,     d=0.0,                            alpha=np.deg2rad(0)),
    RevoluteDH(a=0.0,     d=0.0,                            alpha=np.deg2rad(90)),
    RevoluteDH(a=0.0,     d=-(l0_y_Humerus),                alpha=np.deg2rad(-90)),
    RevoluteDH(a=-l0_x_Humerus, d=(l0_z_Ulna - l0_z_Humerus), alpha=np.deg2rad(-90)),
    RevoluteDH(a=0.0,     d=-(l0_y_Ulna + l0_y_Radius + l0_y1_Radius),     alpha=np.deg2rad(-90)),
    RevoluteDH(a=l0_a_Radius, d= -l0_d_Radius,                        alpha=np.deg2rad(97.8)),
    RevoluteDH(a=0.0,     d=0.0,                            alpha=np.deg2rad(-145.0)),
    RevoluteDH(a=-0.08,   d=0,                              alpha=np.deg2rad(-90)) 
], name='HumanArm')

u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)
sphere = np.array([x.flatten(), y.flatten(), z.flatten()])

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.25, bottom=0.45)

ellip_plot = [ax.plot_surface(x, y, z, color='skyblue', alpha=0.7)]
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Endpoint Stiffness Ellipsoid')

slider_axes = [
    plt.axes([0.25, 0.37 - i*0.035, 0.65, 0.03]) for i in range(8)
]

s_rot_slider = Slider(slider_axes[0], 'shoulder rotation', -90, 90, valinit= 0) #can actually be more than +/- 90
s_aa_slider = Slider(slider_axes[1], 'shoulder elevation', 0, 90, valinit= 0) #elevation/abduction-adduction
s_ax_slider = Slider(slider_axes[2], 'humerus rotation', -45, 90, valinit= 0) #confirm negative limit
elbow_slider = Slider(slider_axes[3], 'elbow flexion', -180, 180, valinit=0)
Ke_slider = Slider(slider_axes[4], 'Elbow Stiffness', 0.1, 200, valinit=100) #units for stiffness are Nm/rad
Ks_ax_slider = Slider(slider_axes[5], 'Shoulder Axial', 0.1, 500, valinit=100)
Ks_aa_slider = Slider(slider_axes[6], 'Shoulder AA', 0.1, 500, valinit=100)
Ks_rot_slider = Slider(slider_axes[7], 'Shoulder Rot', 0.1, 500, valinit=100)

# q = [shoulder rotation, shoulder abduction adduction, shoulder axial rotation, 
#       elbow flexion, radius rotation, wrist deviation, wrist flexion, dummy]
def update(val):
    q = [s_rot_slider.val, s_aa_slider.val, s_ax_slider.val, elbow_slider.val, 0, 101.3, 13.7, -90] # last DOF is just for transforming to thumb
    compute_and_plot(q, Ks_rot_slider.val, Ks_aa_slider.val, Ks_ax_slider.val, Ke_slider.val)
    # robot.plot(q, backend='pyplot', block=False)

for s in [s_rot_slider, s_aa_slider, s_ax_slider, elbow_slider, Ke_slider, Ks_ax_slider, Ks_aa_slider, Ks_rot_slider]:
    s.on_changed(update)

compute_and_plot([0, 0, -90, 180, -124.2, 101.3, 13.7, -90], 1, 1, 1, 1)

plt.show()
