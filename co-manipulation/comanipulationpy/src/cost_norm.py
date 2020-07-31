from metrics import print_metrics
from trajectory_framework import TrajectoryFramework
from traj_calc import TrajectoryPlanner
import numpy as np

def run_single_test(start, target, name='iiwa', traj=131):
    comanipulationFramework = TrajectoryFramework(name, '')
    comanipulationFramework.trajectory_solver.load_traj_file(traj)
    num_timesteps = comanipulationFramework.trajectory_solver.n_pred_timesteps
    coeffs = {
        "nominal": 100.0,
        "distance": [2000.0 for _ in range(num_timesteps)],
        "visibility": [1.5 for _ in range(num_timesteps)],
        "regularize": [7.0 for _ in range(num_timesteps - 1)],
        "legibility": 2000.0,
        "collision": dict(cost=[20], dist_pen=[0.025]),
        "smoothing": dict(cost=300, type=2)
    }
    # coeffs = distance_metric(coeffs)
    comanipulationFramework.scene.robot.SetDOFValues(joint_start, comanipulationFramework.scene.manipulator.GetArmIndices())
    metrics = comanipulationFramework.setup_test(start, target, coeffs, traj_num=traj, execute=False)
    print(metrics)
    return metrics

def distance_metric():
    joint_start = [-0.7240388673767146, -0.34790398102066433, 2.8303899987665897, -2.54032606205873, 1.3329587647643253, 2.7596249683074614, 0.850582268802067]
    joint_target = [-0.3902233335085379, 1.7501020413442578, 0.8403277122861033, -0.22924505085794067, 2.8506926562622024, -1.417026666483551, -0.35668663982214976] #far reaching case
    
    dist_coeff = 2000
    prev_dist_cost = 0.95
    learning_rate = 1000
    iterations = 100
    i = 0
    precision = 0.05
    cost_diff = 0.001
    while cost_diff >= 0 and i < iterations:
        metrics = run_single_test(joint_start, joint_target, name, traj)
        curr_dist_cost = metrics[0]
        cost_diff = curr_dist_cost - prev_dist_cost
        if  cost_diff >= precision:
            learning_rate = learning_rate * 0.9
        dist_coeff +=  learning_rate
        prev_dist_cost = curr_dist_cost
        i += 1
    print(dist_coeff, prev_dist_cost)


if __name__ == "__main__":
    distance_metric()