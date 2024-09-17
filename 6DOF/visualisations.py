# Import necessary modules
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d as plt3d
import matplotlib.cm as cm
matplotlib.use('TkAgg')
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.gridspec as gridspec
from kinematics import kinematics as k
import copy
import math
import numpy as np
import warnings

class visualisations:

    def plot_chain_global(pos_a, ori_a, dir_graph, title=''): #KEEP
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        fig.suptitle(title, fontsize=12)

        x = []
        y = []
        z = []
        # plot chain a
        for i in range(len(pos_a)):
            x.append(pos_a[i][0])
            y.append(pos_a[i][1])
            z.append(pos_a[i][2])

            dirxlocal = ori_a[i] @ [1, 0, 0]
            dirylocal = ori_a[i] @ [0, 1, 0]
            dirzlocal = ori_a[i] @ [0, 0, 1]

            ax.quiver(x[i], y[i], z[i], dirxlocal[0], dirxlocal[1], dirxlocal[2], length=0.05, color='r')
            ax.quiver(x[i], y[i], z[i], dirylocal[0], dirylocal[1], dirylocal[2], length=0.05, color='g')
            ax.quiver(x[i], y[i], z[i], dirzlocal[0], dirzlocal[1], dirzlocal[2], length=0.05, color='b')

        ax.scatter(x, y, z, c='gray',s=100, marker='o')

        parent_child_pairs = k.parent_child(dir_graph)
        for j in range(len(parent_child_pairs)):
            xs = [x[parent_child_pairs[j][0]], x[parent_child_pairs[j][1]]]
            ys = [y[parent_child_pairs[j][0]], y[parent_child_pairs[j][1]]]
            zs = [z[parent_child_pairs[j][0]], z[parent_child_pairs[j][1]]]

            line = art3d.Line3D(xs, ys, zs, linewidth=8, c='black')
            ax.add_line(line)

        ax.add_line(art3d.Line3D([0, pos_a[0][0]], [0, pos_a[0][1]], [0, pos_a[0][2]], c='gray', alpha=0.6, linestyle='--'))

        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)

        ax.set_xlabel('Global X')
        ax.set_ylabel('Global Y')
        ax.set_zlabel('Global Z')

        ax.set_axis_off()

        plt.show()

    def plot_pose_local_aniplusslider(pos_b, dir_graph, start, stop, title=''):
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle(title, fontsize=12)

        ax_slider_plot = fig.add_subplot(121, projection='3d')
        ax_slider_plot.set_axis_off()

        ax_animation_plot = fig.add_subplot(122, projection='3d')
        ax_animation_plot.set_axis_off()

        def plot_frame(ax, frame, show_frame_number=False):
            ax.clear()

            x1, y1, z1 = pos_b[frame][:, 0], pos_b[frame][:, 1], pos_b[frame][:, 2]
            ax.scatter(x1, y1, z1, c='blue',s=100, marker='o')

            parent_child_pairs = k.parent_child(dir_graph)
            for j in range(len(parent_child_pairs)):
                xs = [x1[parent_child_pairs[j][0]], x1[parent_child_pairs[j][1]]]
                ys = [y1[parent_child_pairs[j][0]], y1[parent_child_pairs[j][1]]]
                zs = [z1[parent_child_pairs[j][0]], z1[parent_child_pairs[j][1]]]
                line = art3d.Line3D(xs, ys, zs,linewidth=6, c='gray',linestyle='--')
                ax.add_line(line)

            ax.set_xlim3d(-0.5, 0.5)
            ax.set_ylim3d(-0.5, 0.5)
            ax.set_zlim3d(-0.5, 0.5)
            ax.set_xlabel('Global X')
            ax.set_ylabel('Global Y')
            ax.set_zlabel('Global Z')
            ax.set_axis_off()

            if show_frame_number:
                ax.text2D(0.05, 0.95, f"Frame: {frame}", transform=ax.transAxes)

        def update_slider(val):
            frame = int(val)
            plot_frame(ax_slider_plot, frame)

        ax_frame_slider = plt.axes([0.15, 0.02, 0.3, 0.03], facecolor='lightgoldenrodyellow')
        frame_slider = widgets.Slider(ax_frame_slider, 'Frame', start, stop - 1, valinit=start, valfmt='%0.0f')
        frame_slider.on_changed(update_slider)

        ani = [None]  # Encapsulate the animation in a list to allow modification

        def create_animation(interval):
            if ani[0]:
                ani[0].event_source.stop()  # Stop the previous animation
            ani[0] = FuncAnimation(fig, lambda frame: plot_frame(ax_animation_plot, frame, show_frame_number=True), frames=range(start, stop), interval=interval, repeat=True)

        def update_interval(val):
            interval = int(val)
            create_animation(interval)

        ax_interval_slider = plt.axes([0.6, 0.02, 0.3, 0.03], facecolor='lightgoldenrodyellow')
        interval_slider = widgets.Slider(ax_interval_slider, 'Interval (ms)', 10, 1000, valinit=500, valfmt='%0.0f ms')
        interval_slider.on_changed(update_interval)

        update_slider(start)
        create_animation(500)  # Initialize the animation with the default interval

        plt.show()

    def plot_chain_global_frames_aniplusslider(pos_as, ori_as, dir_graph, start, stop, title=''):
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle(title, fontsize=12)

        ax_slider_plot = fig.add_subplot(121, projection='3d')
        ax_slider_plot.set_axis_off()

        ax_animation_plot = fig.add_subplot(122, projection='3d')
        ax_animation_plot.set_axis_off()

        def plot_frame(ax, frame):
            ax.clear()

            pos_a = pos_as[frame]
            ori_a = ori_as[frame]
            
            x = [p[0] for p in pos_a]
            y = [p[1] for p in pos_a]
            z = [p[2] for p in pos_a]

            # Plot chain
            ax.scatter(x, y, z, c='gray', s=100, marker='o')

            # Plot lines between parent and child
            parent_child_pairs = k.parent_child(dir_graph)
            for j in range(len(parent_child_pairs)):
                xs = [x[parent_child_pairs[j][0]], x[parent_child_pairs[j][1]]]
                ys = [y[parent_child_pairs[j][0]], y[parent_child_pairs[j][1]]]
                zs = [z[parent_child_pairs[j][0]], z[parent_child_pairs[j][1]]]
                line = art3d.Line3D(xs, ys, zs,linewidth=8, c='black')
                ax.add_line(line)

            # Plot local coordinate systems
            for i in range(len(pos_a)):
                dirxlocal = ori_a[i] @ [1, 0, 0]
                dirylocal = ori_a[i] @ [0, 1, 0]
                dirzlocal = ori_a[i] @ [0, 0, 1]
                ax.quiver(x[i], y[i], z[i], dirxlocal[0], dirxlocal[1], dirxlocal[2], length=0.05, color='r')
                ax.quiver(x[i], y[i], z[i], dirylocal[0], dirylocal[1], dirylocal[2], length=0.05, color='g')
                ax.quiver(x[i], y[i], z[i], dirzlocal[0], dirzlocal[1], dirzlocal[2], length=0.05, color='b')

            # Display frame number in the animation plot
            if ax == ax_animation_plot:
                ax.text2D(0.05, 0.95, f"Frame: {frame}", transform=ax.transAxes)

            ax.set_xlim3d(-0.5, 0.5)
            ax.set_ylim3d(-0.5, 0.5)
            ax.set_zlim3d(-0.5, 0.5)
            ax.set_xlabel('Global X')
            ax.set_ylabel('Global Y')
            ax.set_zlabel('Global Z')
            ax.set_axis_off()

        def update_slider(val):
            frame = int(val)
            plot_frame(ax_slider_plot, frame)

        ax_frame_slider = plt.axes([0.15, 0.02, 0.3, 0.03], facecolor='lightgoldenrodyellow')
        frame_slider = widgets.Slider(ax_frame_slider, 'Frame', start, stop - 1, valinit=start, valfmt='%0.0f')
        frame_slider.on_changed(update_slider)

        ani = [None]  # Encapsulate the animation in a list to allow modification

        def create_animation(interval):
            if ani[0]:
                ani[0].event_source.stop()  # Stop the previous animation
            ani[0] = FuncAnimation(fig, lambda frame: plot_frame(ax_animation_plot, frame), frames=range(start, stop), interval=interval, repeat=True)

        def update_interval(val):
            interval = int(val)
            create_animation(interval)

        ax_interval_slider = plt.axes([0.6, 0.02, 0.3, 0.03], facecolor='lightgoldenrodyellow')
        interval_slider = widgets.Slider(ax_interval_slider, 'Interval (ms)', 10, 1000, valinit=500, valfmt='%0.0f ms')
        interval_slider.on_changed(update_interval)

        update_slider(start)
        create_animation(500)  # Initialize the animation with the default interval

        plt.show()



    def plot_chain_interactive(Kchain, dir_graph, dir_graph_pose, bds, pos_b, title='', vector_pairs=[]):
        global global_eas
        global_eas = []

        warnings.filterwarnings('ignore', category=UserWarning)

        # Create a larger figure to accommodate everything comfortably
        fig = plt.figure(figsize=(18, 10))  # Adjust size as needed
        fig.suptitle(title, fontsize=12)

        # Adjust subplot for the 3D plot to be more centrally located
        # Using a GridSpec for the 3D plot as well for consistent layout control
        gs_main = fig.add_gridspec(1, 8)  # 1 row, 8 cols for main layout
        ax = fig.add_subplot(gs_main[0, 1:7], projection='3d')  # Allocate more central cols to 3D plot

        total_sliders = len(bds) * 3  # 3 sliders per joint
        sliders_per_side = total_sliders // 2 + total_sliders % 2

        # Define gridspecs for sliders on both sides of the 3D plot
        left_gs = gridspec.GridSpec(sliders_per_side, 1, figure=fig, left=0.05, right=0.15, wspace=0.05)
        right_gs = gridspec.GridSpec(sliders_per_side, 1, figure=fig, left=0.85, right=0.95, wspace=0.05)

        sliders = []
        for i, joint_bds in enumerate(bds):
            for j, (min_val, max_val) in enumerate(joint_bds):
                axis_label = ['X', 'Y', 'Z'][j]
                joint_label = f"Joint {i} {axis_label}"
                if i * 3 + j < sliders_per_side:
                    ax_slider = fig.add_subplot(left_gs[i * 3 + j])
                else:
                    ax_slider = fig.add_subplot(right_gs[(i * 3 + j) - sliders_per_side])
                slider = widgets.Slider(ax=ax_slider, label=joint_label, valmin=min_val, valmax=max_val, valinit=0, valstep=math.pi/360, valfmt='%1.2f')
                sliders.append(slider)


        # Button for resetting sliders
        ax_reset_all_button = fig.add_axes([0.4, 0.01, 0.2, 0.05])#fig.add_subplot(gs[19, 6:12])
        reset_all_button = widgets.Button(ax_reset_all_button, 'Reset all', hovercolor='1')

        # Function to update kinematic chain visualization based on slider values
        def update_chain(val):
            EAs = [slider.val for slider in sliders]

            # Update kinematic chain
            EAs = np.reshape(EAs,(len(Kchain),3))
            Kchain_reori = k.FK_MDH_reori(copy.deepcopy(Kchain), EAs)
            ori_a, pos_a, _ = k.FK_MDH(Kchain_reori, dir_graph)
            ax.clear()

            # Plotting kinematic chain
            x, y, z = [], [], []
            for i in range(len(pos_a)):
                x.append(pos_a[i][0])
                y.append(pos_a[i][1])
                z.append(pos_a[i][2])
                dirxlocal = ori_a[i] @ [1, 0, 0]
                dirylocal = ori_a[i] @ [0, 1, 0]
                dirzlocal = ori_a[i] @ [0, 0, 1]
                ax.quiver(x[i], y[i], z[i], dirxlocal[0], dirxlocal[1], dirxlocal[2], length=0.05, color='r')
                ax.quiver(x[i], y[i], z[i], dirylocal[0], dirylocal[1], dirylocal[2], length=0.05, color='g')
                ax.quiver(x[i], y[i], z[i], dirzlocal[0], dirzlocal[1], dirzlocal[2], length=0.05, color='b')
                # plot indices
                ax.text(x[i]+0.03, y[i]+0.03, z[i]+0.03,  '%s' % (str(i)), size=8, zorder=1,color='black')
            ax.scatter(x, y, z, c='gray', s=100, marker='o')

            # Connect parent-child pairs with lines
            parent_child_pairs = k.parent_child(dir_graph)
            for j in range(len(parent_child_pairs)):
                xs = [x[parent_child_pairs[j][0]], x[parent_child_pairs[j][1]]]
                ys = [y[parent_child_pairs[j][0]], y[parent_child_pairs[j][1]]]
                zs = [z[parent_child_pairs[j][0]], z[parent_child_pairs[j][1]]]
                line = plt3d.art3d.Line3D(xs, ys, zs, linewidth=8, c='black')
                ax.add_line(line)

            frame = 0  # Example: starting with the first frame
            x1, y1, z1 = pos_b[frame][:, 0], pos_b[frame][:, 1], pos_b[frame][:, 2]
            ax.scatter(x1, y1, z1, c='blue',s=100, marker='o')
            for i in range(len(x1)):
                ax.text(x1[i]+0.03, y1[i]+0.03, z1[i]+0.03,  '%s' % (str(i)), size=8, zorder=1,color='blue')

            # Connect pose estimation points
            parent_child_pairs_pose = k.parent_child(dir_graph_pose)
            for j in range(len(parent_child_pairs_pose)):
                xs = [x1[parent_child_pairs_pose[j][0]], x1[parent_child_pairs_pose[j][1]]]
                ys = [y1[parent_child_pairs_pose[j][0]], y1[parent_child_pairs_pose[j][1]]]
                zs = [z1[parent_child_pairs_pose[j][0]], z1[parent_child_pairs_pose[j][1]]]
                line = plt3d.art3d.Line3D(xs, ys, zs, c='gray',linewidth=6, alpha=0.7, linestyle='--')
                ax.add_line(line)

            # Determine the number of vector pairs to establish the color range
            num_pairs = len(vector_pairs)
            color_map = cm.get_cmap('hsv', num_pairs)  # Use HSV colormap for a wide range of colors

            # Plot vector pairs for kinematic chain and 3D pose
            for index, pair in enumerate(vector_pairs):
                kinematic_pair, pose_pair = pair

                # Generate a unique color for this pair using the colormap
                color = color_map(index)

                # Draw line for kinematic chain
                kinematic_start, kinematic_end = pos_a[kinematic_pair[0]], pos_a[kinematic_pair[1]]
                ax.plot([kinematic_start[0], kinematic_end[0]], [kinematic_start[1], kinematic_end[1]], [kinematic_start[2], kinematic_end[2]], color=color, linewidth=2)

                # Draw line for 3D pose
                pose_start, pose_end = pos_b[frame][pose_pair[0]], pos_b[frame][pose_pair[1]]
                ax.plot([pose_start[0], pose_end[0]], [pose_start[1], pose_end[1]], [pose_start[2], pose_end[2]], color=color, linewidth=2, linestyle=':')


            # Setting the plot limits and labels
            ax.set_xlim3d(-0.5, 0.5)
            ax.set_ylim3d(-0.5, 0.5)
            ax.set_zlim3d(-0.5, 0.5)
            ax.set_xlabel('Global X')
            ax.set_ylabel('Global Y')
            ax.set_zlabel('Global Z')
            
            ax.set_axis_off()

            fig.canvas.draw_idle()

        # Function to save and close
        def save_and_close(event):
            global global_eas
            global_eas = [slider.val for slider in sliders]
            plt.close(fig)

        # Save and Close button
        ax_save_close_button = fig.add_axes([0.4, 0.05, 0.2, 0.05])#fig.add_subplot(gs[18, 6:12])
        save_close_button = widgets.Button(ax_save_close_button, 'Save and Close', hovercolor='1')
        save_close_button.on_clicked(save_and_close)

        # Initial call to update_chain to plot the kinematic chain before any slider interaction
        update_chain(None)

        # Connect sliders to update function
        for slider in sliders:
            slider.on_changed(update_chain)

        # Function to reset all sliders
        def reset_all(event):
            for slider in sliders:
                slider.reset()

        reset_all_button.on_clicked(reset_all)

        plt.show()

        return global_eas

