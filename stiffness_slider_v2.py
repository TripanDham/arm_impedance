import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from roboticstoolbox import DHRobot, RevoluteDH
from mpl_toolkits.mplot3d import Axes3D

def pseudo_inv(J, A):
    A_inv = np.linalg.inv(A)
    B = J @ A_inv @ J.T
    return A_inv @ J.T @ np.linalg.inv(B)

def compute_and_plot(q_deg, Ks_rot, Ks_aa, Ks_ax, Ke):
    q = np.deg2rad(q_deg)
    J = robot.jacob0(q)
    Kj = np.diag([
        Ks_rot, Ks_aa, Ks_ax, Ke, 100000
        # 10000, 10000, 10000, 10000  # Fixed high wrist stiffness
    ])
    J_plus = pseudo_inv(J[:3, :], Kj)
    Ke_matrix = J_plus.T @ Kj @ J_plus
    eigvals, eigvects = np.linalg.eigh(Ke_matrix)
    
    print(Ke_matrix)
    print("Eigenvect", [np.linalg.norm(eigvects[:,i]) for i in range(3)])
    print("Eigvals", eigvals)
    # axes_lengths = 1/np.sqrt(eigvals)

    axes_lengths = eigvals #is sqrt needed?

    # print("Axes lengths: ", axes_lengths)

    ellipsoid = eigvects @ np.diag(axes_lengths) @ sphere #is this correct? do we need to normalise the eigvect
    x_e, y_e, z_e = ellipsoid.reshape(3, *x.shape)

    ellip_plot[0].remove()

    ellip_plot[0] = ax.plot_surface(x_e, y_e, z_e, color='skyblue', alpha=0.7)
    x_e_flat = x_e.flatten()
    y_e_flat = y_e.flatten()
    z_e_flat = z_e.flatten()

    xy_plot[0].remove()
    yz_plot[0].remove()
    zx_plot[0].remove()

    xy_plot[0] = ax2.plot(x_e_flat, y_e_flat, 'b.',markersize = 1)[0]
    yz_plot[0] = ax3.plot(y_e_flat, z_e_flat, 'g.',markersize = 1)[0]
    zx_plot[0] = ax4.plot(z_e_flat, x_e_flat, 'r.',markersize = 1)[0]

    ax.set_xlim([-2000, 2000])
    ax.set_ylim([-2000, 2000])
    ax.set_zlim([-2000, 2000])

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

# In the final frame, the x axis points outward from palm, and the y axis is in the direction of the forearm
robot = DHRobot([
    RevoluteDH(a=0.0,     d=0.0,                            alpha=np.deg2rad(-90)),
    RevoluteDH(a=0.0,     d=0.0,                            alpha=np.deg2rad(90)),
    RevoluteDH(a=0.0,     d=-(l0_y_Humerus),                alpha=np.deg2rad(-90)),
    RevoluteDH(a=-l0_x_Humerus, d=(l0_z_Ulna - l0_z_Humerus), alpha=np.deg2rad(-90)),
    RevoluteDH(a=0.0,     d=-(l0_y_Ulna + l0_y_Radius + l0_y1_Radius),     alpha=np.deg2rad(-90)),
    # RevoluteDH(a=l0_a_Radius, d= -l0_d_Radius,                        alpha=np.deg2rad(90.0)),
    # RevoluteDH(a=0.0,     d=0.0,                            alpha=np.deg2rad(-90.0))
    # RevoluteDH(a=-0.08,   d=0,                              alpha=np.deg2rad(-90)) 
], name='HumanArm')

u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)
sphere = np.array([x.flatten(), y.flatten(), z.flatten()])
x_flat = x.flatten()
y_flat = y.flatten()
z_flat = z.flatten()

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(141, projection='3d')
plt.subplots_adjust(left=0.25, bottom=0.45)

ellip_plot = [ax.plot_surface(x, y, z, color='skyblue', alpha=0.7)]
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Endpoint Stiffness Ellipsoid')

ax2 = fig.add_subplot(142)
xy_plot = [ax2.plot(x_flat, y_flat, 'b.',markersize = 1)[0]]
ax2.set_title("XY projection")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.axis("equal")
ax2.grid(True)

ax3 = fig.add_subplot(143)
yz_plot = [ax3.plot(y_flat, z_flat, 'g.',markersize = 1)[0]]
ax3.set_title("YZ projection")
ax3.set_xlabel("Y")
ax3.set_ylabel("Z")
ax3.axis("equal")
ax3.grid(True)

ax4 = fig.add_subplot(144)
zx_plot = [ax4.plot(z_flat, x_flat, 'r.',markersize = 1)[0]]
ax4.set_title("ZX projection")
ax4.set_xlabel("Z")
ax4.set_ylabel("X")
ax4.axis("equal")
ax4.grid(True)

slider_axes = [
    plt.axes([0.25, 0.37 - i*0.035, 0.65, 0.03]) for i in range(8)
]

s_rot_slider = Slider(slider_axes[0], 'shoulder rotation', -45, 90, valinit= 0) #can actually be more than +/- 90
s_aa_slider = Slider(slider_axes[1], 'shoulder elevation', 0, 90, valinit= 0) #elevation/abduction-adduction
s_ax_slider = Slider(slider_axes[2], 'humerus rotation', -90, 90, valinit= 0) 
elbow_slider = Slider(slider_axes[3], 'elbow flexion', 0, 130, valinit=0)
Ke_slider = Slider(slider_axes[4], 'Elbow Stiffness', 0.1, 50, valinit=5) #units for stiffness are Nm/rad
Ks_ax_slider = Slider(slider_axes[5], 'Shoulder Axial', 0.1, 10, valinit=5)
Ks_aa_slider = Slider(slider_axes[6], 'Shoulder AA', 0.1, 10, valinit=5)
Ks_rot_slider = Slider(slider_axes[7], 'Shoulder Rot', 0.1, 10, valinit=5)

# q = [shoulder rotation, shoulder abduction adduction, shoulder axial rotation, 
#       elbow flexion, radius rotation, wrist deviation, wrist flexion, dummy]

Kss = [Ks_rot_slider.val, Ks_aa_slider.val, Ks_ax_slider.val]
alpha = 0.5
Ke = alpha * np.linalg.norm(Kss)
def update(val):
    q = np.deg2rad([90 + s_rot_slider.val, s_aa_slider.val, -90 + s_ax_slider.val, 180 - elbow_slider.val, 0])
    compute_and_plot(q, Ks_rot_slider.val, Ks_aa_slider.val, Ks_ax_slider.val, Ke_slider.val)
    robot.plot(q, backend='pyplot', block=False, eeframe=True, shadow=True,jointaxes=False,limits=[-0.4,0.4,-0.7,0.7,-0.4,0.4])

for s in [s_rot_slider, s_aa_slider, s_ax_slider, elbow_slider, Ke_slider, Ks_ax_slider, Ks_aa_slider, Ks_rot_slider]:
    s.on_changed(update)

plt.show()
