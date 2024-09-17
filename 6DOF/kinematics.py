# Import necessary modules
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import copy
import scipy.optimize
import warnings

class kinematics:

    def jnt_path(graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            return path
        if not start in graph:
            return None
        for node in graph[start]:
            if node not in path:
                newpath = kinematics.jnt_path(graph, node, end, path)
                if newpath: return newpath
        return None

    def takeSecond(elem):
        return elem[1]

    def parent_child(graph):
        parent_child_pairs = []
        for i in range(max(graph[max(graph)])+1):
            for j in range(max(graph[max(graph)])+1):
                pair = kinematics.jnt_path(graph, i, j)
                if pair != None: 
                    if len(pair)==2:
                        parent_child_pairs.append(pair)
        parent_child_pairs.sort(key=kinematics.takeSecond)
        return parent_child_pairs

    def FK_MDH(chain,dir_graph): 
        """ perform forward kinematics to to convert joint orientations and position vectors into the global coordinate system"""
        oris=[]
        poss=[]
        poss_rel=[]
        for i in range (len(chain)):
            path = kinematics.jnt_path(dir_graph, 0, i)
            T=np.array([[ 1, 0, 0, 0],
                        [ 0, 1, 0, 0],
                        [ 0, 0, 1, 0],
                        [ 0, 0, 0, 1]])
            for j in path:
                T= T @ chain[j]
                pos=np.array([T[0][3],T[1][3],T[2][3]])
                ori=np.array([[T[0][0],T[0][1],T[0][2]],[T[1][0],T[1][1],T[1][2]],[T[2][0],T[2][1],T[2][2]]])
            oris.append(ori)
            poss.append(pos)
        for i in range(len(chain)):
            path = kinematics.jnt_path(dir_graph, 0, i)
            if i==0:
                pos_rel=poss[i]
            else:
                pos_rel=poss[path[-1]]-poss[path[-2]]
            poss_rel.append(pos_rel)
        return oris, poss, poss_rel

    def Rev_FK_MDH(oris, poss, dir_graph): 
        """ reverse operation of forward kinematics to to convert global joint orientations and position vectors back into Denavit-Hartenberg convention"""
        Ts=[]
        for i in range (len(oris)):
            path = kinematics.jnt_path(dir_graph, 0, i)
            path= path[::-1] # reverse path
            if len(path)==1:
                ori= oris[path[0]]
                pos= poss[path[0]]
            else:
                tuple = [path[0], path[1]]
                ori= oris[tuple[1]].T @ oris[tuple[0]]
                pos= oris[tuple[1]].T @ (poss[tuple[0]] - poss[tuple[1]])
            Ts.append(np.array([[ ori[0][0], ori[0][1], ori[0][2], pos[0]],
                    [ ori[1][0], ori[1][1], ori[1][2], pos[1]],
                    [ ori[2][0], ori[2][1], ori[2][2], pos[2]],
                    [ 0, 0, 0, 1]]))
        return Ts

    def EA_XYZ(W1, W2, W3):
        θx = W1
        θy = W2
        θz = W3
        ROTMx = np.array([[1, 0, 0],
                        [0, np.cos(θx), -np.sin(θx)],
                        [0, np.sin(θx), np.cos(θx)]])
        ROTMy = np.array([[np.cos(θy), 0, np.sin(θy)],
                        [0, 1, 0],
                        [-np.sin(θy), 0, np.cos(θy)]])
        ROTMz = np.array([[np.cos(θz), -np.sin(θz), 0],
                        [np.sin(θz), np.cos(θz), 0],
                        [0, 0, 1]])
        
        rotation_matrix = ROTMz @ ROTMy @ ROTMx  # Correct multiplication order
        return rotation_matrix

    def FK_MDH_reori(Kchain, EAs):
        # Ensure EAs is iterable and has sets of 3 values (Euler Angles) for each joint
        if not all(len(ea) == 3 for ea in EAs):
            raise ValueError("Each joint must have 3 Euler angles.")

        # Loop through each set of Euler angles and corresponding Kchain element
        for i, ea in enumerate(EAs):
            # Calculate joint reorientation using Euler angles
            jnt_reori = kinematics.EA_XYZ(*ea)  # Unpack ea list to function arguments
            
            # Create transformation matrix from jnt_reori
            trans_matrix = np.array([
                [jnt_reori[0][0], jnt_reori[0][1], jnt_reori[0][2], 0],
                [jnt_reori[1][0], jnt_reori[1][1], jnt_reori[1][2], 0],
                [jnt_reori[2][0], jnt_reori[2][1], jnt_reori[2][2], 0],
                [0, 0, 0, 1]
            ])

            # Apply transformation to the kinematic chain segment
            Kchain[i] = Kchain[i] @ trans_matrix

        return Kchain


    def XYZ(W1, W2, W3):
        θx = W1 * (2*math.pi)
        θy = W2 * (2*math.pi)
        θz = W3 * (2*math.pi)
        ROTMx = np.array([[1, 0, 0],
                        [0, np.cos(θx), -np.sin(θx)],
                        [0, np.sin(θx), np.cos(θx)]])
        ROTMy = np.array([[np.cos(θy), 0, np.sin(θy)],
                        [0, 1, 0],
                        [-np.sin(θy), 0, np.cos(θy)]])
        ROTMz = np.array([[np.cos(θz), -np.sin(θz), 0],
                        [np.sin(θz), np.cos(θz), 0],
                        [0, 0, 1]])
        
        rotation_matrix = ROTMz @ ROTMy @ ROTMx 
        return rotation_matrix

    def angular_difference(vector1, vector2):
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        if norm1 == 0 or norm2 == 0:
            return 0  # avoid div by zero - if any vector is zero
        cosine_angle = np.clip(np.dot(vector1, vector2) / (norm1 * norm2), -1.0, 1.0)
        return np.arccos(cosine_angle)

    def IK_vector_mapping(Kchain, pose, dir_graph, weights, vector_pairs):
        Local_reoris = []

        # ensure weights can be grouped into sets of 3 values (Euler Angles) for each joint
        if len(weights) % 3 != 0:
            raise ValueError("Weights array length must be a multiple of 3.")

        # calculate the number of joints based on weights length
        num_joints = len(Kchain)

        for i in range(num_joints):
            # extract weights for current joint (3 consecutive values starting from index i*3)
            current_weights = weights[i*3:i*3+3]

            # calculate joint reorientation using extracted weights as Euler angles
            jnt_reori = kinematics.XYZ(*current_weights)

            # create transformation matrix from jnt_reori
            trans_matrix = np.array([
                [jnt_reori[0][0], jnt_reori[0][1], jnt_reori[0][2], 0],
                [jnt_reori[1][0], jnt_reori[1][1], jnt_reori[1][2], 0],
                [jnt_reori[2][0], jnt_reori[2][1], jnt_reori[2][2], 0],
                [0, 0, 0, 1]
            ])

            # apply transformation to the kinematic chain segment
            Kchain[i] = Kchain[i] @ trans_matrix
            Local_reoris.append(R.from_matrix(jnt_reori).as_quat())

        Kchain_global_ori,Kchain_global_pos,Kchain_global_pos_rel = kinematics.FK_MDH(Kchain,dir_graph)

        Errors=[]
        for (kin_start, kin_end), (pose_start, pose_end) in vector_pairs:
            # define vectors in the kinematic chain and pose estimate
            vector_kin = Kchain_global_pos[kin_end] - Kchain_global_pos[kin_start]
            vector_pose = pose[pose_end] - pose[pose_start]

            # calculate the angular difference between vectors
            angle_diff = kinematics.angular_difference(vector_kin, vector_pose)
            Errors.append(angle_diff)

        return Kchain, Local_reoris, Kchain_global_ori,Kchain_global_pos,Errors

    def loss_pose(weights, chain, pose, dir_graph,vector_pairs):
        _,_,_,_,E = kinematics.IK_vector_mapping(copy.deepcopy(chain), pose, dir_graph, weights,vector_pairs)
        SE = [i**2 for i in E]
        MSE = np.mean(np.nansum(SE))
        return MSE
    
    def IK_opt_frames(Kchain,poses,dir_graph,bds, initial_weights,vector_pairs):

        results_of_minimizations=[]
        for frame in range(len(poses)): 
            if frame == 0:
                weights = initial_weights
                warnings.filterwarnings('ignore', 'Values in x were outside bounds during a minimize step, clipping to bounds')
            else:
                weights = results_of_minimizations[-1].x
            pose=poses[frame]

            result_of_minimization = scipy.optimize.minimize(kinematics.loss_pose, weights, method='SLSQP', bounds=bds, args=(copy.deepcopy(Kchain),pose,dir_graph,vector_pairs))

            results_of_minimizations.append(result_of_minimization)

            # update progress
            print(f"Frame {frame}/{len(poses)-1}", end='\r')
            print()

        print()
        return results_of_minimizations

    def loss_dEAs(weights, frames):
        dEAs = []
        for frame in range(frames):          
            if frame == 0:  # forward difference
                dEAs.append(np.abs(np.array(weights[frame + 1]) - np.array(weights[frame])))#
            elif frame < frames - 1:  # central difference
                dEAs.append(np.abs((np.array(weights[frame + 1]) - np.array(weights[frame - 1])) / 2))
            else:  # backward difference
                dEAs.append(np.abs(np.array(weights[frame]) - np.array(weights[frame - 1])))#
        return dEAs

    def IK_vector_mapping_temporal(Kchain_in, poses, dir_graph, weights_all, temp_weight,vector_pairs):
        Errors = []
        num_joints = len(Kchain_in)
        num_frames = len(poses)
        weights_all = np.reshape(weights_all,(num_frames,num_joints*3))

        for frame in range(len(poses)):
            Kchain = copy.deepcopy(Kchain_in)
            weights = weights_all[frame]  # extract weights for the current frame
            Local_reoris = []
            pose = poses[frame]

            for i in range(num_joints):
                # extract weights for the current joint in the current frame
                current_weights = weights[i*3:i*3+3]

                # calculate joint reorientation using extracted weights as Euler angles
                jnt_reori = kinematics.XYZ(*current_weights)

                # create transformation matrix from jnt_reori
                trans_matrix = np.array([
                    [jnt_reori[0][0], jnt_reori[0][1], jnt_reori[0][2], 0],
                    [jnt_reori[1][0], jnt_reori[1][1], jnt_reori[1][2], 0],
                    [jnt_reori[2][0], jnt_reori[2][1], jnt_reori[2][2], 0],
                    [0, 0, 0, 1]
                ])

                # apply transformation to the kinematic chain segment
                Kchain[i] = Kchain[i] @ trans_matrix
                Local_reoris.append(R.from_matrix(jnt_reori).as_quat())
            Kchain_global_ori,Kchain_global_pos,Kchain_global_pos_rel = kinematics.FK_MDH(Kchain,dir_graph)

            for (kin_start, kin_end), (pose_start, pose_end) in vector_pairs:
                # define vectors in the kinematic chain and pose estimate
                vector_kin = Kchain_global_pos[kin_end] - Kchain_global_pos[kin_start]
                vector_pose = pose[pose_end] - pose[pose_start]

                # calculate the angular difference between vectors
                angle_diff = kinematics.angular_difference(vector_kin, vector_pose)
                Errors.append(angle_diff)

        dEAs = kinematics.loss_dEAs(weights,len(poses))
        dEAs = np.array(dEAs).flatten('F')
        for i in range(len(dEAs)):
            Errors.append(temp_weight*dEAs[i])

        return Kchain, Local_reoris, Kchain_global_ori,Kchain_global_pos,Errors

    def loss_pose_temporal(weights, chain, pose, dir_graph,temp_weight,vector_pairs): # check whether the pose errors are different w/w/o dEAs errors
        _,_,_,_,E = kinematics.IK_vector_mapping_temporal(copy.deepcopy(chain), pose, dir_graph, weights,temp_weight,vector_pairs)
        SE = [i**2 for i in E]
        MSE = np.mean(np.nansum(SE))
        return MSE
    
    def IK_opt_frames_temporal(Kchain, poses, dir_graph, bds, temp_weight, initial_weights,vector_pairs):
        result_of_minimization = scipy.optimize.minimize(kinematics.loss_pose_temporal, initial_weights, method='SLSQP', bounds=bds, args=(copy.deepcopy(Kchain), poses, dir_graph, temp_weight,vector_pairs), options={'maxiter': 2000, 'disp': True})
        # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html#optimize-minimize-slsqp
        #methods that alse work well: 'Powell', 'CG', ‘BFGS’ (fastest and works for ME - Powell does not) 'L-BFGS-B' or 'SLSQP' and the latter seems to work better and more consistently
        print(result_of_minimization)
        print()
        return result_of_minimization