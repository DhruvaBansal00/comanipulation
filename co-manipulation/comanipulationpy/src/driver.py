from trajectory_framework import TrajectoryFramework
from tests import Test
from metrics import print_metrics, metrics_to_csv, save_experiments
import numpy as np
from plots import show_radar_chart

def analyze_multiple_trajectories(trajectories, joint_start, joint_target, execute_comanipulation, execute_baseline, 
                    num_baselines, num_metrics):
    comanipulationFramework = TrajectoryFramework(robot, '')
    comanipulationFramework.scene.robot.SetDOFValues(joint_start, comanipulationFramework.scene.manipulator.GetArmIndices())

    all_comanipulation_metrics = np.zeros((num_metrics, len(trajectories)))
    all_baseline_metrics = np.zeros((num_baselines, num_metrics, len(trajectories)))

    for trajIndex, trajectory in enumerate(trajectories):
        all_comanipulation_metrics[:, trajIndex] = comanipulationFramework.setup_test(joint_start, joint_target, traj_num=trajectory, execute=execute_comanipulation)
        baselineTest = Test(robot,joint_start, joint_target, traj_num=trajectory, execute=execute_baseline)
        all_baseline_metrics[:, :, trajIndex] = baselineTest.run_all_baselines()
        metrics_to_csv(trajectory, all_comanipulation_metrics[:, trajIndex], all_baseline_metrics[:, :, trajIndex])
    print_metrics(all_comanipulation_metrics, all_baseline_metrics)
    plot_multiple_trajectories(all_comanipulation_metrics, all_baseline_metrics)
    test_case = ''.join(str(test) + ", " for test in trajectories)
    save_experiments(test_case[:-2], all_comanipulation_metrics, all_baseline_metrics)

def plot_multiple_trajectories(comanip_metrics, other_metrics):
    other_metric_order = ["Distance + Visibility Baseline", "Legibility Baseline", "Nominal Trajectory Baseline", "Speed Control Baseline"]
    our_means = [np.mean(x) for x in comanip_metrics]
    other_means = [[np.mean(x) for x in bl] for bl in other_metrics]
    show_radar_chart(our_means, zip(other_metric_order, other_means))

def simple_interp(start, target, name='jaco'):
    comanipulationFramework = TrajectoryFramework(name, '')
    comanipulationFramework.scene.robot.SetDOFValues(joint_start, comanipulationFramework.scene.manipulator.GetArmIndices())
    comanipulationFramework.setup_ros()
    raw_input("Ready to move to start")
    comanipulationFramework.scene.follow_joint_trajectory_client.move_to(start)
    raw_input("Ready to move to target")
    comanipulationFramework.scene.follow_joint_trajectory_client.move_to(target)


def run_single_test(start, target, name='jaco', traj=303):
    comanipulationFramework = TrajectoryFramework(name, '')
    comanipulationFramework.scene.robot.SetDOFValues(joint_start, comanipulationFramework.scene.manipulator.GetArmIndices())
    comanipulationFramework.setup_test(start, target, traj_num=traj, execute=True)
    

if __name__ == "__main__":
    ################## PARAMETERS ################
    #iiwa
    joint_start = [-0.7240388673767146, -0.34790398102066433, 2.8303899987665897, -2.54032606205873, 1.3329587647643253, 2.7596249683074614, 0.850582268802067]
    joint_target = [-0.21084585626752084, 1.696737816218337, -2.565970219832999, 0.17682048063096367, 2.5144914879697158, 1.2588615840260928, -0.1733579520497237]

    #jaco
    # joint_start = [0.31, 4.63, -1.76, 4.56, -1.58, 4.64, 0]
    # joint_target = [3.1, 4.63, -1.78, 4.56, -3.1, 4.64, 0]

    # simple_interp(joint_start, joint_target)
    # run_single_test(joint_start, joint_target, name='iiwa')
    

    
    trajectories = [120, 303]
    execute_comanipulation = False
    execute_baseline = False
    robot = 'iiwa'
    collision_threshold = 0.25
    num_baselines = 4
    num_metrics = 4
    ###############################################
    analyze_multiple_trajectories(trajectories, joint_start, joint_target, execute_comanipulation, execute_baseline,
          num_baselines, num_metrics)
