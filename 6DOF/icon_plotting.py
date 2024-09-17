""" Implementation of the methods described in https://kevgildea.github.io/blog/Euler-Axis-Vector-Mapping/"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
import math

# functions
# ----------------------------------------------------------------------------------------------------------------
def Vector_mapping_cross_product(vec1, vec2):
    """ Calculate the rotation matrix that maps vector a to align with vector b about an axis aligned with the cross product"""
    a, b = (vec1 / np.linalg.norm(vec1)), (vec2 / np.linalg.norm(vec2))
    n = np.cross(a, b) / np.linalg.norm(np.cross(a, b))
    adotb=np.dot(a, b)
    rotation_matrix = np.array([[n[0]**2+(n[1]**2+n[2]**2)*(adotb),n[0]*n[1]*(1-adotb)-n[2]*np.linalg.norm(np.cross(a, b)),n[0]*n[2]*(1-adotb)+n[1]*np.linalg.norm(np.cross(a, b))],
                                [n[0]*n[1]*(1-adotb)+n[2]*np.linalg.norm(np.cross(a, b)),n[1]**2+(n[0]**2+n[2]**2)*(adotb),n[1]*n[2]*(1-adotb)-n[0]*np.linalg.norm(np.cross(a, b))],
                                [n[0]*n[2]*(1-adotb)-n[1]*np.linalg.norm(np.cross(a, b)),n[1]*n[2]*(1-adotb)+n[0]*np.linalg.norm(np.cross(a, b)),n[2]**2+(n[0]**2+n[1]**2)*(adotb)]])
    return rotation_matrix


def Vector_mapping_bisect(vec1, vec2):
    """ Calculate the rotation matrix that maps vector a to align with vector b about an axis aligned with norm(a) + norm(b)"""
    a, b = (vec1 / np.linalg.norm(vec1)), (vec2 / np.linalg.norm(vec2))
    n = (a+b) / np.linalg.norm(a+b)
    θ = math.pi
    rotation_matrix = np.array([[n[0]**2+(n[1]**2+n[2]**2)*(np.cos(θ)),n[0]*n[1]*(1-np.cos(θ))-n[2]*np.sin(θ),n[0]*n[2]*(1-np.cos(θ))+n[1]*np.sin(θ)],
                                [n[0]*n[1]*(1-np.cos(θ))+n[2]*np.sin(θ),n[1]**2+(n[0]**2+n[2]**2)*(np.cos(θ)),n[1]*n[2]*(1-np.cos(θ))-n[0]*np.sin(θ)],
                                [n[0]*n[2]*(1-np.cos(θ))-n[1]*np.sin(θ),n[1]*n[2]*(1-np.cos(θ))+n[0]*np.sin(θ),n[2]**2+(n[0]**2+n[1]**2)*(np.cos(θ))]])
    return rotation_matrix


def Vector_mapping_Euler_Axis_Space(vec1, vec2): 
    """ Calculate all rotation matrices that map vector a to align with vector b"""
    a, b = (vec1 / np.linalg.norm(vec1)), (vec2 / np.linalg.norm(vec2))
    p1 = np.array([0,0,0])
    p2 = b / np.linalg.norm(b) + (a / np.linalg.norm(a))
    p3 = np.cross(a, b) / np.linalg.norm(np.cross(a, b))
    # create a list of candidate Euler axes (discritised to 1 degree)
    n=[]
    for i in range(0,360,1):
        α=np.radians(i)
        x_α=p1[0]+np.cos(α)*(p2[0]-p1[0])+np.sin(α)*(p3[0]-p1[0])
        y_α=p1[1]+np.cos(α)*(p2[1]-p1[1])+np.sin(α)*(p3[1]-p1[1])
        z_α=p1[2]+np.cos(α)*(p2[2]-p1[2])+np.sin(α)*(p3[2]-p1[2])
        n_α=[x_α,y_α,z_α]
        n.append(n_α / np.linalg.norm(n_α))
    # project vectors to form a cone around the Euler axis, and determine required angle for mapping
    rotation_matrices=[]
    Euler_axes=[]
    Euler_angles=[]
    for i in range(len(n)):
        Euler_axes.append(n[i])
        a_α, b_α = (np.cross(a,n[i]) / np.linalg.norm(np.cross(a,n[i]))), (np.cross(b,n[i]) / np.linalg.norm(np.cross(b,n[i])))
        θ = np.arccos(np.dot(a_α,b_α))
        θ = θ*np.sign(np.dot(n[i], np.cross(a_α,b_α)))
        Euler_angles.append(θ)
        if θ != θ: # if θ is NaN
            rotation_matrices.append(np.array([[ 1, 0, 0],
                                               [ 0, 1, 0],
                                               [ 0, 0, 1]]))
        else:
            rotation_matrices.append(np.array([[n[i][0]**2+(n[i][1]**2+n[i][2]**2)*(np.cos(θ)),n[i][0]*n[i][1]*(1-np.cos(θ))-n[i][2]*np.sin(θ),n[i][0]*n[i][2]*(1-np.cos(θ))+n[i][1]*np.sin(θ)],
                                               [n[i][0]*n[i][1]*(1-np.cos(θ))+n[i][2]*np.sin(θ),n[i][1]**2+(n[i][0]**2+n[i][2]**2)*(np.cos(θ)),n[i][1]*n[i][2]*(1-np.cos(θ))-n[i][0]*np.sin(θ)],
                                               [n[i][0]*n[i][2]*(1-np.cos(θ))-n[i][1]*np.sin(θ),n[i][1]*n[i][2]*(1-np.cos(θ))+n[i][0]*np.sin(θ),n[i][2]**2+(n[i][0]**2+n[i][1]**2)*(np.cos(θ))]]))
  
    return Euler_axes, Euler_angles, rotation_matrices
# ----------------------------------------------------------------------------------------------------------------


# choose vectors a and b
# ----------------------------------------------------------------------------------------------------------------
a=[0.1,0.3,0.4]
b=[0.3,0.1,0.2]
# ----------------------------------------------------------------------------------------------------------------


# Plot vectors a and b
# ----------------------------------------------------------------------------------------------------------------
fig = plt.figure(0)
ax = fig.add_subplot(111, projection='3d')
fig.suptitle('Vectors a and b', fontsize=12)

axb=np.cross(a, b)

ax.quiver(0,0,0,a[0],a[1],a[2],color='black')
ax.text(a[0],a[1],a[2],  '%s' % (r'$\vec a$'), size=12, zorder=1,color='black')
ax.quiver(0,0,0,b[0],b[1],b[2],color='grey')
ax.text(b[0],b[1],b[2],  '%s' % (r'$\vec b$'), size=12, zorder=1,color='grey')
ax.quiver(0,0,0,axb[0],axb[1],axb[2],color='orange',linestyle='--')
ax.text(axb[0],axb[1],axb[2],  '%s' % (r'$\vec {axb}$'), size=12, zorder=1,color='orange')

dirxglobal=[1,0,0]
diryglobal=[0,1,0]
dirzglobal=[0,0,1]

ax.quiver(-0.5,-0.5,-0.5,dirxglobal[0],dirxglobal[1],dirxglobal[2],color='r')
ax.quiver(-0.5,-0.5,-0.5,diryglobal[0],diryglobal[1],diryglobal[2],color='g')
ax.quiver(-0.5,-0.5,-0.5,dirzglobal[0],dirzglobal[1],dirzglobal[2],color='b')

ax.set_xlim3d(-1,1)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(-1,1)

ax.set_xlabel('Global X')
ax.set_ylabel('Global Y')
ax.set_zlabel('Global Z')

plt.show()

# ----------------------------------------------------------------------------------------------------------------


# Plot rotation to map vectors about axis aligned with the cross product
# ----------------------------------------------------------------------------------------------------------------
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
fig.suptitle('Rotation to map vectors about axis aligned with the cross product', fontsize=12)

ax.quiver(0,0,0,a[0],a[1],a[2],color='black')
ax.text(a[0],a[1],a[2],  '%s' % (r'$\vec a$'), size=12, zorder=1,color='black')
ax.quiver(0,0,0,b[0],b[1],b[2],color='grey')
ax.text(b[0],b[1],b[2],  '%s' % (r'$\vec b$'), size=12, zorder=1,color='grey')
ax.quiver(0,0,0,axb[0],axb[1],axb[2],color='orange',linestyle='--')
ax.text(axb[0],axb[1],axb[2],  '%s' % (r'$\vec {axb}$'), size=12, zorder=1,color='orange')

a_mapped = Vector_mapping_cross_product(a, b) @ a
ax.quiver(0,0,0,a_mapped[0],a_mapped[1],a_mapped[2],color='black',linestyle='--')
ax.text(a_mapped[0],a_mapped[1],a_mapped[2],  '%s' % (r'$\vec a_{mapped}$'), size=12, zorder=1,color='black')

θ = np.arccos((np.trace(Vector_mapping_cross_product(a, b))-1)/2)

dirxglobal=[1,0,0]
diryglobal=[0,1,0]
dirzglobal=[0,0,1]

ax.quiver(-0.5,-0.5,-0.5,dirxglobal[0],dirxglobal[1],dirxglobal[2],color='r')
ax.quiver(-0.5,-0.5,-0.5,diryglobal[0],diryglobal[1],diryglobal[2],color='g')
ax.quiver(-0.5,-0.5,-0.5,dirzglobal[0],dirzglobal[1],dirzglobal[2],color='b')

n = (axb / np.linalg.norm(axb)).reshape(3)
ax.quiver(-0.5,-0.5,-0.5,n[0],n[1],n[2],color='orange',linestyle='--')
ax.text(n[0]-0.5,n[1]-0.5,n[2]-0.5,  '%s' % (r'$\vec {n}$'+',  θ='+ str(round(θ, 5))), size=12, zorder=1,color='orange')

dirxglobal=Vector_mapping_cross_product(a, b) @ [1,0,0]
diryglobal=Vector_mapping_cross_product(a, b) @ [0,1,0]
dirzglobal=Vector_mapping_cross_product(a, b) @ [0,0,1]

ax.quiver(-0.5,-0.5,-0.5,dirxglobal[0],dirxglobal[1],dirxglobal[2],color='r',linestyle='--')
ax.quiver(-0.5,-0.5,-0.5,diryglobal[0],diryglobal[1],diryglobal[2],color='g',linestyle='--')
ax.quiver(-0.5,-0.5,-0.5,dirzglobal[0],dirzglobal[1],dirzglobal[2],color='b',linestyle='--')

ax.set_xlim3d(-1,1)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(-1,1)

ax.set_xlabel('Global X')
ax.set_ylabel('Global Y')
ax.set_zlabel('Global Z')

plt.show()
# ----------------------------------------------------------------------------------------------------------------


# Plot rotation to map vectors about axis bisecting vectors a and b
# ----------------------------------------------------------------------------------------------------------------
fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
fig.suptitle('Rotation to map vectors about axis bisecting vectors a and b', fontsize=12)

# choose a rotation axis in the plane bisecting a and b i.e. where a and b are symmetrical
axis = b / np.linalg.norm(b) + a / np.linalg.norm(a)

ax.quiver(0,0,0,a[0],a[1],a[2],color='black')
ax.text(a[0],a[1],a[2],  '%s' % (r'$\vec a$'), size=12, zorder=1,color='black')
ax.quiver(0,0,0,b[0],b[1],b[2],color='grey')
ax.text(b[0],b[1],b[2],  '%s' % (r'$\vec b$'), size=12, zorder=1,color='grey')
ax.quiver(0,0,0,axis[0],axis[1],axis[2],color='orange',linestyle='--')
ax.text(axis[0],axis[1],axis[2],  '%s' % (r'$\vec {b+a}$'), size=12, zorder=1,color='orange')

a_mapped = Vector_mapping_bisect(a, b) @ a
ax.quiver(0,0,0,a_mapped[0],a_mapped[1],a_mapped[2],color='black',linestyle='--')
ax.text(a_mapped[0],a_mapped[1],a_mapped[2],  '%s' % (r'$\vec a_{mapped}$'), size=12, zorder=1,color='black')

θ = np.arccos((np.trace(Vector_mapping_bisect(a, b))-1)/2)

dirxglobal=[1,0,0]
diryglobal=[0,1,0]
dirzglobal=[0,0,1]

ax.quiver(-0.5,-0.5,-0.5,dirxglobal[0],dirxglobal[1],dirxglobal[2],color='r')
ax.quiver(-0.5,-0.5,-0.5,diryglobal[0],diryglobal[1],diryglobal[2],color='g')
ax.quiver(-0.5,-0.5,-0.5,dirzglobal[0],dirzglobal[1],dirzglobal[2],color='b')

n = (axis / np.linalg.norm(axis)).reshape(3)
ax.quiver(-0.5,-0.5,-0.5,n[0],n[1],n[2],color='orange',linestyle='--')
ax.text(n[0]-0.5,n[1]-0.5,n[2]-0.5,  '%s' % (r'$\vec {n}$'+',  θ='+ str(round(θ, 5))), size=12, zorder=1,color='orange')

dirxglobal=Vector_mapping_bisect(a, b) @ [1,0,0]
diryglobal=Vector_mapping_bisect(a, b) @ [0,1,0]
dirzglobal=Vector_mapping_bisect(a, b) @ [0,0,1]

ax.quiver(-0.5,-0.5,-0.5,dirxglobal[0],dirxglobal[1],dirxglobal[2],color='r',linestyle='--')
ax.quiver(-0.5,-0.5,-0.5,diryglobal[0],diryglobal[1],diryglobal[2],color='g',linestyle='--')
ax.quiver(-0.5,-0.5,-0.5,dirzglobal[0],dirzglobal[1],dirzglobal[2],color='b',linestyle='--')

ax.set_xlim3d(-1,1)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(-1,1)

ax.set_xlabel('Global X')
ax.set_ylabel('Global Y')
ax.set_zlabel('Global Z')

plt.show()
# ----------------------------------------------------------------------------------------------------------------


# Plot possible rotation axis space - any vector on the bisecting symmetric plane for a and b
# ----------------------------------------------------------------------------------------------------------------
fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
fig.suptitle('Possible rotation axis space - any vector on the bisecting symmetric plane for a and b', fontsize=12)

# i.e. create the plane which symmetrically bisects vectors a and b
p1 = np.array([0,0,0])
p2 = b / np.linalg.norm(b) + (a / np.linalg.norm(a))
p3 = np.cross(a, b)

cp = np.cross(p2, p3)

#print('The equation is {0}x + {1}y + {2}z = {3}'.format(cp[0], cp[1], cp[2], np.dot(cp, p3)))
ax.text(0,0.7,0.7,  '{0}x + {1}y + {2}z = {3}'.format(round(cp[0],3), round(cp[1],3), round(cp[2],3), round(np.dot(cp, p3),3)), size=10, zorder=1,color='grey')

x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)
Z = (np.dot(cp, p3) - cp[0] * X - cp[1] * Y) / cp[2]

ax.plot(X.flatten(),
        Y.flatten(),
        Z.flatten(), 'bo', marker='.', alpha=0.3)

ax.plot(*zip(p1, p2, p3), color='r', linestyle=' ', marker='o')

ax.quiver(0,0,0,a[0],a[1],a[2],color='black')
ax.text(a[0],a[1],a[2],  '%s' % (r'$\vec a$'), size=12, zorder=1,color='black')
ax.quiver(0,0,0,b[0],b[1],b[2],color='grey')
ax.text(b[0],b[1],b[2],  '%s' % (r'$\vec b$'), size=12, zorder=1,color='grey')

dirxglobal=[1,0,0]
diryglobal=[0,1,0]
dirzglobal=[0,0,1]

ax.quiver(-0.5,-0.5,-0.5,dirxglobal[0],dirxglobal[1],dirxglobal[2],color='r')
ax.quiver(-0.5,-0.5,-0.5,diryglobal[0],diryglobal[1],diryglobal[2],color='g')
ax.quiver(-0.5,-0.5,-0.5,dirzglobal[0],dirzglobal[1],dirzglobal[2],color='b')

ax.set_xlim3d(-1,1)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(-1,1)

ax.set_xlabel('Global X')
ax.set_ylabel('Global Y')
ax.set_zlabel('Global Z')


plt.show()
# ----------------------------------------------------------------------------------------------------------------


# Plot the possible Euler axis space - any unit vector on the bisecting symmetric plane for a and b
# ----------------------------------------------------------------------------------------------------------------
fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')
fig.suptitle('Possible Euler axis space - any unit vector on the bisecting symmetric plane for a and b', fontsize=12)
p1 = np.array([0,0,0])
p2 = b / np.linalg.norm(b) + (a / np.linalg.norm(a))
p3 = np.cross(a, b) / np.linalg.norm(np.cross(a, b))
s1 = np.linspace(-1, 1, 100) 
s2 = np.linspace(-1, 1, 100)
axes=[]
for i in range(0,360,1):
    Φ=np.radians(i)
    x_Φ=p1[0]+np.cos(Φ)*(p2[0]-p1[0])+np.sin(Φ)*(p3[0]-p1[0])
    y_Φ=p1[1]+np.cos(Φ)*(p2[1]-p1[1])+np.sin(Φ)*(p3[1]-p1[1])
    z_Φ=p1[2]+np.cos(Φ)*(p2[2]-p1[2])+np.sin(Φ)*(p3[2]-p1[2])
    n=[x_Φ,y_Φ,z_Φ]
    axes.append(n / np.linalg.norm(n))

print(np.shape(axes))
ax.plot(*zip(p1, p2, p3), color='r', linestyle=' ', marker='o')

for i in range(0,len(axes),2):
    ax.quiver(0,0,0,axes[i][0],axes[i][1],axes[i][2],color='orange',linestyle='--',alpha=0.3)

ax.quiver(0,0,0,a[0],a[1],a[2],color='black')
ax.text(a[0],a[1],a[2],  '%s' % (r'$\vec a$'), size=12, zorder=1,color='black')
ax.quiver(0,0,0,b[0],b[1],b[2],color='grey')
ax.text(b[0],b[1],b[2],  '%s' % (r'$\vec b$'), size=12, zorder=1,color='grey')

dirxglobal=[1,0,0]
diryglobal=[0,1,0]
dirzglobal=[0,0,1]

ax.quiver(-0.5,-0.5,-0.5,dirxglobal[0],dirxglobal[1],dirxglobal[2],color='r')
ax.quiver(-0.5,-0.5,-0.5,diryglobal[0],diryglobal[1],diryglobal[2],color='g')
ax.quiver(-0.5,-0.5,-0.5,dirzglobal[0],dirzglobal[1],dirzglobal[2],color='b')

ax.set_xlim3d(-1,1)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(-1,1)

ax.set_xlabel('Global X')
ax.set_ylabel('Global Y')
ax.set_zlabel('Global Z')


plt.show()
# ----------------------------------------------------------------------------------------------------------------


# Plot rotation to map vectors about all axes in the possible Euler axis space
# ----------------------------------------------------------------------------------------------------------------
fig = plt.figure(5)
ax = fig.add_subplot(111, projection='3d')
fig.suptitle('Demonstration of valid mapping for all Euler axes on the plane', fontsize=12)

Euler_axes, Euler_angles, rotation_matrices = Vector_mapping_Euler_Axis_Space(a, b)

ax.quiver(0,0,0,a[0],a[1],a[2],color='black')
ax.text(a[0],a[1],a[2],  '%s' % (r'$\vec a$'), size=12, zorder=1,color='black')
ax.quiver(0,0,0,b[0],b[1],b[2],color='grey')
ax.text(b[0],b[1],b[2],  '%s' % (r'$\vec b$'), size=12, zorder=1,color='grey')

dirxglobal=[1,0,0]
diryglobal=[0,1,0]
dirzglobal=[0,0,1]

ax.quiver(-0.5,-0.5,-0.5,dirxglobal[0],dirxglobal[1],dirxglobal[2],color='r')
ax.quiver(-0.5,-0.5,-0.5,diryglobal[0],diryglobal[1],diryglobal[2],color='g')
ax.quiver(-0.5,-0.5,-0.5,dirzglobal[0],dirzglobal[1],dirzglobal[2],color='b')

for i in range(0,len(rotation_matrices),2):
    a_mapped = rotation_matrices[i] @ a
    #ax.quiver(0,0,0,a_mapped[0],a_mapped[1],a_mapped[2],color='black',linestyle='--',alpha=0.3)
    #ax.text(a_mapped[0],a_mapped[1],a_mapped[2],  '%s' % (r'$\vec a_{mapped}$'), size=12, zorder=1,color='black',alpha=0.3)
    θ = Euler_angles[i]

    n = Euler_axes[i]
    ax.quiver(-0.5,-0.5,-0.5,n[0],n[1],n[2],color='orange',linestyle='--', alpha=0.3)

    dirxglobal=rotation_matrices[i] @ [1,0,0]
    diryglobal=rotation_matrices[i] @ [0,1,0]
    dirzglobal=rotation_matrices[i] @ [0,0,1]

    ax.quiver(-0.5,-0.5,-0.5,dirxglobal[0],dirxglobal[1],dirxglobal[2],color='r',linestyle='--', alpha=0.3)
    ax.quiver(-0.5,-0.5,-0.5,diryglobal[0],diryglobal[1],diryglobal[2],color='g',linestyle='--', alpha=0.3)
    ax.quiver(-0.5,-0.5,-0.5,dirzglobal[0],dirzglobal[1],dirzglobal[2],color='b',linestyle='--', alpha=0.3)

ax.set_xlim3d(-1,1)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(-1,1)

ax.set_xlabel('Global X')
ax.set_ylabel('Global Y')
ax.set_zlabel('Global Z')
ax.set_axis_off()

plt.show()
# ----------------------------------------------------------------------------------------------------------------