"""The template of the main script of the machine learning process
"""

import pickle
import numpy as np
from os import listdir
from os.path import isfile, join

dirpath = 'C:\\Users\\User\\Student\\MLGame-beta4.0\\games\\arkanoid\\log'
BallPosition = []
PlatformPosition = []
LRUP = []
last_ball_x = 0
last_ball_y = 0
files = listdir(dirpath)
log_number = 0
for k in range(0,8):
    for f in files:
      log_number = log_number + 1
      fullpath = join(dirpath, f)
      if isfile(fullpath):
        with open(fullpath , "rb") as f1:
            data_list1 = pickle.load(f1)
        for i in range(0 , len(data_list1)):
            BallPosition.append(data_list1[i].ball)
            PlatformPosition.append(data_list1[i].platform)
            if(i>=-1):
                if(last_ball_x - data_list1[i].ball[0] > 0):
                    LR = 1
                else:
                    LR = 0
                if(last_ball_y - data_list1[i].ball[1] > 0):
                    UP = 0
                else:
                    UP = 1
                LRUP.append(np.array((LR,UP)))
            last_ball_x = data_list1[i].ball[0]
            last_ball_y = data_list1[i].ball[1]


PlatX = np.array(PlatformPosition) [:,0][:,np.newaxis]
PlatX_next = PlatX[1:,:]
instrust = (PlatX_next-PlatX[0:len(PlatX_next),0][:,np.newaxis])/5

Ballarray = np.array(BallPosition[:-1])
LRUP = np.array((LRUP[:-1]))
x = np.hstack((Ballarray,LRUP,PlatX[0:-1,0][:,np.newaxis]))


y = instrust 
np.set_printoptions(threshold=np.inf)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 41)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn = KNeighborsClassifier(n_neighbors = 11)
knn.fit(x_train,y_train)

yknn_bef_scaler = knn.predict(x_test)
acc_knn_bef_scaler = accuracy_score(yknn_bef_scaler,y_test)

print('log number : ' + str(log_number))
print(acc_knn_bef_scaler)


filename = "C:\\Users\\User\\Student\\MLGame-beta4.0\\games\\arkanoid\\ml\\knn_example.sav"
pickle.dump(knn,open(filename,"wb"))
    