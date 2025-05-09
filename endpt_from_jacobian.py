from roboticstoolbox import DHRobot, RevoluteDH
import numpy as np
import matplotlib.pyplot as plt


def pseudo_inv(J, A):
    A_inv = np.linalg.inv(A)
    B = J @ A_inv @ np.transpose(J)

    return A_inv @ np.transpose(J) @ np.linalg.inv(B)


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
    RevoluteDH(a=l0_a_Radius, d= -l0_d_Radius,                        alpha=np.deg2rad(90.0)),
    RevoluteDH(a=0.0,     d=0.0,                            alpha=np.deg2rad(-90.0))
    # RevoluteDH(a=-0.08,   d=0,                              alpha=np.deg2rad(-90)) 
], name='HumanArm')


# Joint configuration (example values in radians)
# q = np.deg2rad([0, 0, -90, 180, -124.2, 101.3, 13.7, -90])
elbow_angle = 20
q = np.deg2rad([0,0,0, 180 - elbow_angle,0,0,0]) 
#wrist rotation negative rotates hand clockwise about radius


# Compute the Jacobian
J = robot.jacob0(q)
print("Jacobian (6x7) in base frame:\n", J)

# Plot the arm
robot.plot(q, block=True,jointaxes=False, ) #red - x, green - y, blue - z
#x - downwards palm
#y - along forearm length outwards
plt.show()

Ks_rot = 1 #rotator cuff stiffness
Ks_aa = 1 #abduction adduction stiffness
Ks_ax = 1 #axial rotation stiffness (rotation about humerus)
Ke = 1 #elbow stiffness
Kw_rot = 10000 #wrist rotation stiffness
Kw_dev = 10000 #wrist deviation stiffness
Kw_flex = 10000 #wrist flexion extension stiffness

Kj = np.diag([Ks_rot, Ks_aa, Ks_ax, Ke, Kw_rot, Kw_flex, Kw_dev])
#wrist set to very high stiffness now. Confirm if flex before dev or vice vers

J_plus = pseudo_inv(J[:3, :], Kj)

Ke = J_plus.T @ Kj @ J_plus

# U, S, VT = np.linalg.svd(Ke)
eigvals, eigvects = np.linalg.eigh(Ke)
axes_lengths = 1/np.sqrt(eigvals)

# Parametric unit sphere
u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)

# Stack into (3, N) array
sphere = np.array([x.flatten(), y.flatten(), z.flatten()])

# Transform sphere to ellipsoid
ellipsoid = eigvects @ np.diag(axes_lengths) @ sphere
x_e, y_e, z_e = ellipsoid.reshape(3, *x.shape)

# Plot
# fig = plt.figure(2)
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x_e, y_e, z_e, color='skyblue', alpha=0.7)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('3D Endpoint Stiffness Ellipsoid')
# plt.show()
