import numpy as np
import traj_utils 
import scene_utils
import traj_file_utils
import trajopt_request_utils as req_util
import metrics
import plots
import json
import trajoptpy
# trajoptpy.SetInteractive(True)
import trajoptpy.math_utils as mu
import random

from scipy.interpolate import CubicSpline

import time

ADD_RANDOM_CONFIG = False
CHECK_SYMMETRIC_TRAJ = False

class TrajectoryPlanner:
    def __init__(self, scene, n_human_joints=11, n_robot_joints=7):
        self.scene = scene
        self.n_human_joints = n_human_joints
        self.n_robot_joints = n_robot_joints

    def optimize_problem(self, request):
        """
        Returns the result of running trajopt on a problem specified in `request`

        request: A JSON-formatted solver request
        """
        s = json.dumps(request)
        # Create object that stores optimization problem
        prob = trajoptpy.ConstructProblem(s, self.scene.env)
        t_start = time.time()
        result = trajoptpy.OptimizeProblem(prob)  # do optimization
        t_elapsed = time.time() - t_start
        print(result)
        print("optimization took %.3f seconds" % t_elapsed)

        # from trajoptpy.check_traj import traj_is_safe
        # prob.SetRobotActiveDOFs() # set robot DOFs to DOFs in optimization problem
        # assert traj_is_safe(result.GetTraj(), self.scene.robot) # Check that trajectory is collision free

        return result

    def get_default_traj(self, init_joint, final_joint, num_timesteps):
        """
        Returns a trajectory constrained only by joint velocities and the
        corresponding end effector cartesian trajectory

        init_joint: the initial set of joint angles
        final_joint: the desired set of joint angles 
        """
        self.scene.robot.SetDOFValues(init_joint, self.scene.manipulator.GetArmIndices())

        request = req_util.create_empty_request(
            num_timesteps, final_joint, self.scene.manipulator_name)

        req_util.add_joint_vel_cost(request, 1)

        result = self.optimize_problem(request)
        eef_traj = self.scene.follow_trajectory(np.array(result.GetTraj()))

        return result.GetTraj(), eef_traj

    def load_traj_file(self, traj_num):
        """
        Loads a trajectory file and stores the trajectory in instance variables for later use.
        Also stores number of timesteps in self.n_pred_timesteps.

        traj_num: the number of the trajectory to load
        """
        self.full_rightarm_test_traj, self.obs_rightarm_test_traj, self.complete_pred_traj_means, self.complete_pred_traj_vars = traj_file_utils.load_all_human_trajectories(
            traj_num)
        self.complete_pred_traj_means_expanded, self.complete_pred_traj_vars_expanded = traj_utils.expand_human_pred(
            self.complete_pred_traj_means, self.complete_pred_traj_vars)
        self.n_pred_timesteps = len(
            self.complete_pred_traj_means_expanded) / (self.n_human_joints * 3)
        
        self.full_complete_test_traj = traj_utils.create_human_plot_traj(self.full_rightarm_test_traj)
        self.obs_complete_test_traj = traj_utils.create_human_plot_traj(self.obs_rightarm_test_traj)
        self.num_human_timesteps = len(self.full_complete_test_traj) / (self.n_human_joints * 3)
        self.final_obs_timestep_ind = len(self.obs_complete_test_traj) / (self.n_human_joints * 3)
        head_ind = 5
        torso_ind = 6
        self.head_pos = self.full_complete_test_traj[(self.final_obs_timestep_ind * self.n_human_joints + head_ind) * 3 : (self.final_obs_timestep_ind * self.n_human_joints + head_ind + 1) * 3]
        self.torso_pos = self.full_complete_test_traj[(self.final_obs_timestep_ind * self.n_human_joints + torso_ind) * 3 : (self.final_obs_timestep_ind * self.n_human_joints + torso_ind + 1) * 3]
        self.feet_pos = [self.torso_pos[0], self.torso_pos[1], self.torso_pos[2] - 0.5]

    def set_traj(self, complete_traj_means, complete_traj_vars):
        """
        Accepts predicted human trajectories and sets the variables necessary to solve
        Also stores number of predicted timesteps in self.n_pred_timesteps

        complete_traj_means: the expected human trajectory (complete, not arm-only)
        complete_traj_vars: the covariance matrices associated with complete_traj_means
        """
        self.full_rightarm_test_traj, self.obs_rightarm_test_traj = None, None
        self.complete_pred_traj_means, self.complete_pred_traj_vars = complete_traj_means, complete_traj_vars
        self.complete_pred_traj_means_expanded, self.complete_pred_traj_vars_expanded = traj_utils.expand_human_pred(
            self.complete_pred_traj_means, self.complete_pred_traj_vars)
        self.n_pred_timesteps = len(
            self.complete_pred_traj_means_expanded) / (self.n_human_joints * 3)

    def solve_traj_save_plot_exec(self, init_joint, final_joint, coeffs={}, object_pos=[0, 0.2, 0.83],
                plot='', execute=False, save=''):
        """
        NOTE: THIS IS ONLY COMPATIBLE WITH TRAJECTORIES THAT HAVE BEEN LOADED IN WITH load_traj_file
        A convenience function, which solves a trajectory and then optionally executes, plots, and saves it.

        First four arguments (init_joint, final_joint, coeffs, and object_pos): same as solve_traj
        plot: where to save the plot of the end effector trajectory, empty string to do nothing
        execute: whether to execute the calculated trajectory (boolean)
        save: where to save the trajectory as a text file, empty string to do nothing
        """
        result, eef_traj = self.solve_traj(init_joint, final_joint, coeffs=coeffs, object_pos=object_pos)
        _, default_traj = self.get_default_traj(
            init_joint, final_joint, self.n_pred_timesteps)
        if execute:
            # TODO: this method for timestep calculation should leverage class-level n_joint variables
            self.scene.execute_full_trajectory(result.GetTraj(), self.full_rightarm_test_traj, len(
                self.obs_rightarm_test_traj) / 12, len(self.full_rightarm_test_traj) / 12)
        if plot != '':
            full_complete_test_traj = traj_utils.create_human_plot_traj(
                self.full_rightarm_test_traj)
            plots.plot_trajectory(eef_traj, "Distance", default_traj, "Joint Space Linear",
                                  plot, full_complete_test_traj, 11)
        if save != '':
            np.savetxt(save, eef_traj, delimiter=',')
            
        return result, eef_traj

    def solve_traj(self, init_joint, final_joint, coeffs={}, object_pos=[0, 0.2, 0.83]):
        """
        Calculates an optimal trajectory from init_joint to final_joint based on the weights in coeffs.
        Returns joint trajectory and corresponding end effector trajectory

        init_joint: the initial set of joint angles
        final_joint: the desired set of joint angles
        coeffs: A dictionary containing information on weights. All keys are optional.
        Valid keys are 
            "distance": array of length num_timesteps
            "distanceBaseline": array of length num_timesteps
            "collision": a dictionary mapping 'cost' and 'dist_pen' to number arrays
            "nominal": number
            "regularize": array of length num_timesteps - 1
            "smoothing": dictionary mapping 'cost' and 'type' to a number and an int, respectively
            "velocity": array of length num_timesteps (this is a CoMOTO cost, not the trajopt joint velocity cost)
            "visibility": array of length num_timesteps
            "visibilityBaseline": array of length num_timesteps
            "legibility": number
            "legibilityBaseline": number
            "joint_vel": number or [number] of length 1. This is the trajopt joint velocity cost.
        object_pos: The position of the object of interest to the person. Only needed for
            visiblity cost
        """
        self.scene.robot.SetDOFValues(init_joint, self.scene.manipulator.GetArmIndices())

        _, default_traj = self.get_default_traj(
            init_joint, final_joint, self.n_pred_timesteps)
        self.scene.robot.SetDOFValues(init_joint, self.scene.manipulator.GetArmIndices())

        request = req_util.create_empty_request(
            self.n_pred_timesteps, final_joint, self.scene.manipulator_name)
        if "distance" in coeffs:
            req_util.add_distance_cost(request, self.complete_pred_traj_means_expanded,
                                       self.complete_pred_traj_vars_expanded, coeffs["distance"], self.n_human_joints, self.scene.all_links)
        if "distanceBaseline" in coeffs:
            req_util.add_distance_baseline_cost(request, self.head_pos, self.torso_pos, self.feet_pos, self.scene.all_links, self.n_pred_timesteps, coeffs["distanceBaseline"])
        
        if "visibilityBaseline" in coeffs:
            req_util.add_visibility_baseline_cost(request, self.head_pos, object_pos, self.scene.eef_link_name, self.n_pred_timesteps, coeffs["visibilityBaseline"])

        if "legibilityBaseline" in coeffs:
            req_util.add_legibility_baseline_cost(
                request, coeffs["legibilityBaseline"], self.scene.eef_link_name)
        if "collision" in coeffs:
            req_util.add_collision_cost(
                request, coeffs["collision"]["cost"], coeffs["collision"]["dist_pen"])
        if "nominal" in coeffs:
            req_util.add_optimal_trajectory_cost(
                request, default_traj, self.scene.eef_link_name, self.n_pred_timesteps, coeffs["nominal"])
        if "regularize" in coeffs:
            req_util.add_regularize_cost(
                request, coeffs["regularize"], self.scene.eef_link_name)
        if "smoothing" in coeffs:
            req_util.add_smoothing_cost(
                request, coeffs["smoothing"]["cost"], coeffs["smoothing"]["type"])
        if "velocity" in coeffs:
            req_util.add_velocity_cost(request, self.complete_pred_traj_means_expanded,
                                       self.complete_pred_traj_vars_expanded, coeffs["velocity"], self.n_human_joints, self.scene.all_links)
        if "visibility" in coeffs:
            head_pred_traj_mean, head_pred_traj_var = traj_utils.create_human_head_means_vars(
                self.complete_pred_traj_means_expanded, self.complete_pred_traj_vars_expanded)
            req_util.add_visibility_cost(request, head_pred_traj_mean, head_pred_traj_var,
                                         coeffs["visibility"], object_pos, self.scene.eef_link_name)
        if "legibility" in coeffs:
            req_util.add_legibility_cost(
                request, coeffs["legibility"], self.scene.eef_link_name)
        
        if "joint_vel" in coeffs:
            req_util.add_joint_vel_cost(request, coeffs["joint_vel"])

        if CHECK_SYMMETRIC_TRAJ:
            final_joint = self.get_best_goal(self.scene.robot.GetDOFValues().tolist(), final_joint)
            req_util.set_goal(request, final_joint)
            self.scene.robot.SetDOFValues(init_joint, self.scene.manipulator.GetArmIndices())

        if ADD_RANDOM_CONFIG:
            random_traj = self.get_random_start_traj(self.scene.robot.GetDOFValues().tolist(), final_joint)
            req_util.set_init_traj(request, random_traj)
            self.scene.robot.SetDOFValues(init_joint, self.scene.manipulator.GetArmIndices())

        # raw_input("Enter to continue...")

        result = self.optimize_problem(request)
        eef_traj = self.scene.follow_trajectory(np.array(result.GetTraj()))
        return result, eef_traj

    def get_best_goal(self, start, end):
        alternate_end = [j for j in end]
        base_joint_idx = self.scene.manipulator.GetArmIndices()[0]
        base_joint_limits = self.scene.robot.GetJoints()[base_joint_idx].GetLimits()
        base_joint_min, base_joint_max = base_joint_limits[0][0], base_joint_limits[1][0]

        if start[0] > end[0]:
            while start[0] > alternate_end[0]:
                alternate_end[0] += 2*3.1415926
            if alternate_end[0] > base_joint_max:
                print "Can't check alternative, %0.3f > %0.3f" % (alternate_end[0], base_joint_max)
                return end
        else:
            while start[0] < alternate_end[0]:
                alternate_end[0] -= 2*3.1415926
            if alternate_end[0] < base_joint_min:
                print "Can't check alternative, %0.3f < %0.3f" % (alternate_end[0], base_joint_max)
                return end
        
        default_traj = mu.linspace2d(start, end, self.n_pred_timesteps)
        alternate_traj = mu.linspace2d(start, alternate_end, self.n_pred_timesteps)
        default_traj_score = self.score_traj(default_traj) 
        alternate_traj_score = self.score_traj(alternate_traj)

        print "Default traj score: %0.3f\nAlternative traj score: %0.3f" % (default_traj_score, alternate_traj_score)

        if default_traj_score > alternate_traj_score:
            return end
        return alternate_end

    def get_random_joint_config(self):
        while True:
            cfg = self.get_random_joint_config_helper()
            posn = self.scene.get_eef_position(cfg)
            if posn[2] > 0.1: # this gets a configuration with the eef at least 0.1 m above the ground
                return cfg

    def get_random_joint_config_helper(self):
        ret_val = []
        joint_list = self.scene.robot.GetJoints()
        for joint_idx in self.scene.manipulator.GetArmIndices():
            joint_min = joint_list[joint_idx].GetLimits()[0][0]
            joint_max = joint_list[joint_idx].GetLimits()[1][0]
            val = random.uniform(joint_min, joint_max)
            ret_val.append(val)
        return ret_val

    def get_traj_through_waypoint(self, start, waypoint, end):
        waypoint_step = self.n_pred_timesteps // 2
        inittraj = np.empty((self.n_pred_timesteps, 7))
        inittraj[:waypoint_step+1] = mu.linspace2d(start, waypoint, waypoint_step+1)
        inittraj[waypoint_step:] = mu.linspace2d(waypoint, end, self.n_pred_timesteps - waypoint_step)
        return inittraj

    def score_traj(self, traj):
        n_obs_timesteps = len(self.obs_rightarm_test_traj) / 12
        return 5 * metrics.compute_distance_metric(
            self.scene,
            self.full_complete_test_traj,
            n_obs_timesteps,
            n_obs_timesteps + self.n_pred_timesteps,
            traj
        ) #- 0.05 * sum(np.linalg.norm(traj[i+1] - traj[i]) for i in range(len(traj) - 1))

    def get_random_start_traj(self, start, end, n_config=60):
        best_traj = self.get_traj_through_waypoint(start, self.get_random_joint_config(), end)
        best_metrics = self.score_traj(best_traj)
        for _ in range(n_config):
            curr_traj = self.get_traj_through_waypoint(start, self.get_random_joint_config(), end)
            curr_score = self.score_traj(curr_traj)
            if curr_score > best_metrics:
                best_traj = curr_traj
                best_metrics = curr_score
        return [row.tolist() for row in  best_traj]

    def calculate_adaptive_trajectory(self, robot_joints, human_traj):
        '''
        Takes in a human and a robot trajectory and returns a version of that trajectory in which the
        robot follows the same path but 

        robot_joints: time-sampled JOINT space trajectory (vectorized - timesteps*robot_num_joints*3 matrix)
        human_traj: human position from vision system (vectorized - timesteps*human_num_joints*3matrix)
        the order of human_traj is - right_shoulder + right_elbow + right_wrist + right_palm + neck + head + 
            torso + left_shoulder + left_elbow + left_wrist + left_palm
        ideally, robot_joints >= human_traj_timesteps
        '''
        scaling_factor = 1  
        d_slow = 0.15 # choose threshold
        d_stop = 0.06 # choose threshold
        beta = 3.3332 # parameter
        gamma = 0.5 # parameter
        traj_interpolation = cubic_interpolation(robot_joints, self.n_robot_joints)
        num_timesteps = len(human_traj)/(self.n_human_joints*3)
        robot_total_timesteps = len(robot_joints)
        new_exec_traj = [] 
        robot_timestep = 0
        done = False
        human_timestep = 0
        while not done:
            new_robot_joints = traj_interpolation(robot_timestep).tolist()
            new_exec_traj.append(new_robot_joints)

            curr_human_pos = human_traj[human_timestep*self.n_human_joints*3:(human_timestep + 1)*self.n_human_joints*3]
            d = metrics.get_separation_dist(self.scene, curr_human_pos, new_robot_joints)
            if (d_stop <= d <= d_slow):
                scaling_factor = 1 - beta * ((d - d_stop) ** gamma)
            elif (d_slow < d):# change to new timestamp and trajectory pts
                scaling_factor = 0
            else:
                scaling_factor = 1
            
            robot_timestep += 0.1 - scaling_factor/10
            if human_timestep < num_timesteps - 1:
                human_timestep += 1

            if robot_total_timesteps <= robot_timestep:
                done = True
            elif human_timestep == (num_timesteps - 1) and scaling_factor == 1:
                done = True
            else:
                done = False
        
        return new_exec_traj

def cubic_interpolation(robot_joints, robot_num_joints):
    """
    Returns a scipy CubicSpline object that splines between trajectory waypoints in robot_joints

    robot_joints: An array representing a robot joint trajectory. 1-D or 2-D.
    robot_num_joints: The number of joints the robot has
    """
    robot_joints = np.array(robot_joints)
    robot_joints_pos = np.reshape(robot_joints, (-1, robot_num_joints))
    x = list(range(0, len(robot_joints_pos)))
    return CubicSpline(x, robot_joints_pos)