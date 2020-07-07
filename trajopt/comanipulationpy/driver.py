from trajectory_framework import TrajectoryFramework
from tests import Test
from metrics import print_metrics
import numpy as np


def analyze_multiple_trajectories(trajectories, joint_start, joint_target, execute_comanipulation, execute_baseline, 
                                enable_estop, resume_safely, collision_threshold, num_baselines, num_metrics):
    comanipulationFramework = TrajectoryFramework(robot, '', enable_estop=enable_estop, resume_safely=resume_safely, collision_threshold=collision_threshold)
    comanipulationFramework.scene.robot.SetDOFValues(joint_start, comanipulationFramework.scene.manipulator.GetArmIndices())

    all_comanipulation_metrics = np.zeros((num_metrics, len(trajectories)))
    all_baseline_metrics = np.zeros((num_baselines, num_metrics, len(trajectories)))

    for trajIndex, trajectory in enumerate(trajectories):
        all_comanipulation_metrics[:, trajIndex] = comanipulationFramework.setup_test(joint_start, joint_target, traj_num=trajectory, execute=execute_comanipulation)
        baselineTest = Test(robot,joint_start, joint_target, traj_num=trajectory, execute=execute_baseline, 
                        enable_estop=enable_estop, resume_safely=resume_safely, collision_threshold=collision_threshold)
        all_baseline_metrics[:, :, trajIndex] = baselineTest.run_all_baselines()
    print_metrics(all_comanipulation_metrics, all_baseline_metrics)

def simple_interp(start, target, name='jaco'):
    comanipulationFramework = TrajectoryFramework(name, '')
    comanipulationFramework.scene.robot.SetDOFValues(joint_start, comanipulationFramework.scene.manipulator.GetArmIndices())
    comanipulationFramework.setup_ros()
    raw_input("Ready to move to start")
    comanipulationFramework.scene.follow_joint_trajectory_client.move_to(start)
    raw_input("Ready to move to target")
    comanipulationFramework.scene.follow_joint_trajectory_client.move_to(target)

def run_single_test(start, target, name='jaco'):
    comanipulationFramework = TrajectoryFramework(name, '')
    comanipulationFramework.scene.robot.SetDOFValues(joint_start, comanipulationFramework.scene.manipulator.GetArmIndices())
    comanipulationFramework.setup_test(start, target, traj_num=303, execute=True)

if __name__ == "__main__":

    ################## PARAMETERS ################
    #iiwa
    joint_start = [-0.7240388673767146, -0.34790398102066433, 2.8303899987665897, -2.54032606205873, 1.3329587647643253, 2.7596249683074614, 0.850582268802067]
    joint_target = [-0.21084585626752084, 1.696737816218337, -2.565970219832999, 0.17682048063096367, 2.5144914879697158, 1.2588615840260928, -0.1733579520497237]

    #jaco
    # joint_start = [0.31, 4.63, -1.76, 4.56, -1.58, 4.64, 0]
    # joint_target = [3.1, 4.63, -1.78, 4.56, -3.1, 4.64, 0]
    # simple_interp(joint_start, joint_target)
    run_single_test(joint_start, joint_target, name='iiwa')
    

    
    # trajectories = [120, 303]
    # enable_estop = False
    # resume_safely = False
    # execute_comanipulation = True
    # execute_baseline = False
    # robot = 'iiwa'
    # collision_threshold = 0.25
    # num_baselines = 4
    # num_metrics = 4
    # ###############################################
    # analyze_multiple_trajectories(trajectories, joint_start, joint_target, execute_comanipulation, execute_baseline, enable_estop, 
    #                             resume_safely, collision_threshold, num_baselines, num_metrics)
