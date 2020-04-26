import pandas as pd
import matplotlib.mlab as mlab
import numpy
import matplotlib.pyplot as plt
import math

#расстояние до ближайших
def dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2))

data_train = pd.read_csv("train.csv")
data_validate = pd.read_csv("validate.csv")

MTrain_H = []
MTrain_W = []
WTrain_H = []
WTrain_W = []
MValidate_H = []
MValidate_W = []
WValidate_H = []
WValidate_W = []

for i in range(0, len(data_train)):
    if data_train.iloc[i]['Gender'] == 'Male':
        MTrain_H.append(data_train.iloc[i].Height)
        MTrain_W.append(data_train.iloc[i].Weight)
    if data_train.iloc[i]['Gender'] == 'Female':
        WTrain_H.append(data_train.iloc[i].Height)
        WTrain_W.append(data_train.iloc[i].Weight)

for i in range(0, len(data_validate)):
    #print(data_validate.iloc[i])
    if data_validate.iloc[i]['Gender'] == 'Male':
        MValidate_H.append(data_validate.iloc[i].Height)
        MValidate_W.append(data_validate.iloc[i].Weight)
    if data_validate.iloc[i]['Gender'] == 'Female':
        WValidate_H.append(data_validate.iloc[i].Height)
        WValidate_W.append(data_validate.iloc[i].Weight)

#гистограмма
fig, ax = plt.subplots()
num_bins = 5
n, bins, patches = plt.hist(MTrain_W, num_bins, facecolor='red', alpha=0.5)
plt.show()
n1, bins, patches = plt.hist(WTrain_W, num_bins, facecolor='blue', alpha=0.5)
plt.show()

for _ in range(0, len(MValidate_W)):
    input_w = MValidate_W[_]
    input_h = MValidate_H[_]
    gender = "Male"

   # print("w : ", input_w, " h : ", input_h)
   # input()

    r = 20
    score = 0
    distances = []

    for i in range(0, len(MTrain_H)):
        tmp_dist = []
        tmp_dist.append(dist(input_w, input_h, MTrain_H[i], MTrain_W[i]))
        tmp_dist.append("man")
        distances.append(tmp_dist)

    for i in range(len(WTrain_H)):
        tmp_dist = []
        tmp_dist.append(dist(input_w, input_h, WTrain_H[i], WTrain_W[i]))
        tmp_dist.append("woman")
        distances.append(tmp_dist)

    #print(distances)
    distances.sort(key=lambda tup: tup[0])
    print(distances)

    k = 6000
    #while distances < r:
    mans = 0
    womans = 0
    print("m : ", mans, "w : ", womans)
    for l in range(k):
        if distances[l][1] == "man":
            mans += 1
        else:
            womans += 1
print("m : ", mans, "w : ", womans)
print("man prob:", mans / k * 100)
print("woman prob:", womans / k * 100)
print("================================")
        

