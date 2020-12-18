#! /usr/bin/env python
import rospy
import csv
import matlab.engine
from std_msgs.msg import String
import numpy as np
import ast
import json
import math
from sklearn.metrics import mean_squared_error
import pandas as pd

def callback(data):
    if len(data.data) != 0 and data.data != "Waiting for enough data":
        traj, expData, expSigma = data.data.split("splitTag")
        traj, expData, expSigma = matlab.double(json.loads(str(traj))), matlab.double(json.loads(str(expData))[:100]), matlab.double(json.loads(str(expSigma))[:100])
        predictedMeans, variances = [],[]


        i = 0
#        print(len(expData))
        for row in expData:
            #for col in row:
            if i % 10 == 0:
                predictedMeans.append(row)
            i += 1
        i = 0
 #       print("NUM")

        #rint(np.array(traj).shape)
        for timestep in range(len(expSigma)):
            data = []
            if i % 10 == 0:
                for row in range(len(expSigma[timestep])):
                    for col in expSigma[timestep][row]:
                        data.append(col)
                variances.append(data)
            i+=1
        
        #print("Mean length = " + str(len(predictedMeans)) + " Variance length = " + str(len(variances)))
        df1 = pd.DataFrame(predictedMeans)
        df1.to_csv('predSampledtraj_2610_trimmed.csv',index=False, header=False)
        df2 = pd.DataFrame(variances)
        df2.to_csv('varPredSampledtraj_2610_trimmed.csv',index=False, header=False)
        df3 = pd.DataFrame(traj)
        df3.to_csv('traj_2610_trimmed.csv',index=False,header=False)
        df3.append(df1, ignore_index=True)
        df3.to_csv('traj_2610.csv', index=False,header=False)

        #print(predictedMeans)
        # print(str(len(expSigma)) + " " + str(len(expSigma[0])) + " " + str(len(expSigma[0][0])))
            
def listener():
    rospy.init_node('parse', anonymous=False)
    rospy.Subscriber("human_traj_pred", String, callback)
    rospy.spin()

if __name__ == '__main__':
    print("Calling Listener")
    listener()