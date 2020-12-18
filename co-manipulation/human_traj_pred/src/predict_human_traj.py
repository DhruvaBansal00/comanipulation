#! /usr/bin/env python
import rospy
import csv
import matlab.engine
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray, Marker
import numpy as np
import ast

class Traj:
    def __init__(self, eng, pub):
        self.obsv = []
        self.matlab = eng
        self.pub = pub
        self.pred = None
    
    def add(self, obv):
        self.obsv.append(obv)
    def discard(self):
        self.obsv.pop(0)
    def publish(self):
        self.pub.publish(self.pred)

    def generate_prediction(self):
        shoulder_elbow_wrist_palm = [12,13,14,15]
        traj = []
        for obv in self.obsv:
            curr_frame = [0]
            for index in shoulder_elbow_wrist_palm:
                curr_frame.append(obv[index].pose.position.x)
                curr_frame.append(obv[index].pose.position.y)
                curr_frame.append(obv[index].pose.position.z)
            traj.append(curr_frame)
        expData, expSigma = self.matlab.UOLA_predict('trainedGMM.mat', matlab.double(traj), 'prediction.csv', nargout=2)
        #print(np.array(traj).shape)
        for row in traj:
            del row[0]
        print(np.array(traj).shape)
        self.pred = str(traj) + "splitTag" + str(expData)+"splitTag"+str(expSigma)


def callback(data, curr_obsv):
    curr_obsv.add(data.markers)
    if len(curr_obsv.obsv) > 50:
        curr_obsv.generate_prediction()
        curr_obsv.discard()
            
def listener(eng):
    rospy.init_node('predictor', anonymous=False)
    pub = rospy.Publisher('human_traj_pred', String, queue_size=10)
    curr_obsv = Traj(eng, pub)
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