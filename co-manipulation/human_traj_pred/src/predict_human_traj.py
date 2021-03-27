#! /usr/bin/env python
import rospy
import tf
import csv
import matlab.engine
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray, Marker
import numpy as np
import ast
from math import e
import pandas as pd

class Traj:
    def __init__(self, eng, pub):
        self.obsv = []
        self.matlab = eng
        self.pub = pub
        self.pred = None
        self.transform = tf.transformations.euler_matrix(-2.278, 0.097, 3.125)[:3, :3]#tf.transformations.euler_matrix(-2.185351651204997, 0.0025157588914009534, 2.9853950940245584)[:3, :3] #np.eye(3) #np.array([[0.175213, 0.381727, 0.545808],[-0, 0.389293, 0.344862], [-5.78241*e**-17, 1.33333, 1]])
    
    def add(self, obv):
        self.obsv.append(obv)
    def discard(self):
        self.obsv.pop(0)
    def publish(self):
        self.pub.publish(self.pred)

    def generate_prediction(self):
        shoulder_elbow_wrist_palm = [12,13,14,15] #for right hand https://docs.microsoft.com/en-us/azure/kinect-dk/body-joints
        traj = []
        for obv in self.obsv:
            curr_frame = [0]
            for index in shoulder_elbow_wrist_palm:
                # -0.1380266 , -0.41772856, -1.04825784
                # 0.047, 0.303, 1.181
                old_coords = np.array([obv[index].pose.position.x, obv[index].pose.position.y, obv[index].pose.position.z])
                new_coords = np.matmul(self.transform, old_coords.T)
                curr_frame.append(new_coords[0] + 0.047)#1.4)#
                curr_frame.append(new_coords[1] + 0.303)#0.5)#
                curr_frame.append(new_coords[2] + 1.181)#-0.5)#
            traj.append(curr_frame)
        expData, expSigma = self.matlab.UOLA_predict('trainedGMM.mat', matlab.double(traj), 'prediction.csv', nargout=2)
        for row in traj:
            del row[0]
        df1 = pd.DataFrame(traj)
        df1.to_csv('data_traj_motion.csv',index=False, header=False)
        print(np.array(traj).shape)
        self.pred = str(traj) + "splitTag" + str(expData)+"splitTag"+str(expSigma)


def callback(data, curr_obsv):
    print("HELLOOOO")
    curr_obsv.add(data.markers)
    if len(curr_obsv.obsv) > 50:
        curr_obsv.generate_prediction()
        curr_obsv.discard()
            
def listener(eng):
    rospy.init_node('predictor', anonymous=False)
    pub = rospy.Publisher('human_traj_pred', String, queue_size=10)
    curr_obsv = Traj(eng, pub)
    print("Subscribing")
    rospy.Subscriber("/front/body_tracking_data", MarkerArray, callback, (curr_obsv))
    #/front/body_tracking_data
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        if curr_obsv.pred is not None:
            curr_obsv.publish()
            rate.sleep()

if __name__ == '__main__':
    print("Starting Matlab engine")
    eng = matlab.engine.start_matlab()
    print("Started Matlab. Calling Listner")
    eng.cd('UOLA', nargout=0)
    listener(eng)
