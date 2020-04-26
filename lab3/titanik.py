import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tqdm

data_people = pd.read_csv("data.csv")
#print(len(data_people))


data = data_people.dropna(subset=['Age'])
data.reset_index(drop=True)
#print(len(data))

Woman_A = []
Woman_F = []
Man_A = []
Man_F = []

for i in range(0, len(data)):
    if data.iloc[i]['Sex'] == 'male':
        Man_A.append(data.iloc[i].Age)
        Man_F.append(data.iloc[i].Fare)
    if data.iloc[i]['Sex'] == 'female':
        Woman_A.append(data.iloc[i].Age)
        Woman_F.append(data.iloc[i].Fare)

for j in range(0, len(data)):
    if data.iloc[j]['Survived'] == 0:
        plt.plot(Woman_A, Woman_F, '.', color='red')
    elif data.iloc[j]['Survived'] == 1:
        plt.plot(Woman_A, Woman_F, '*', color='green')
plt.show()

for l in range(0, len(data)):
    if data.iloc[l]['Survived'] == 0:
        plt.plot(Man_A, Man_F, '.', color='red')
    elif data.iloc[l]['Survived'] == 1:
        plt.plot(Man_A, Man_F, '.', color='green')
plt.show()

def distance(a,b):
    d = 0
    d += abs(a['Pclass'] - b['Pclass'])
    d += a['Sex'] != b['Sex']
    d += abs(a['Age'] - b['Age'])
    d += abs(a['SibSp'] - b['SibSp'])
    d += abs(a['Parch'] - b['Parch'])
    d += abs(a['Fare'] - b['Fare'])
    d += a['Embarked'] != b['Embarked']
    return d

def myKNeighborsClassifier(learnData, K, passengerIndexForPrediction):
    dists = np.zeros((learnData.shape[0] - 1, 2))
    i = 0
    for idx, row in learnData.iterrows():
        if idx != passengerIndexForPrediction:  #LOO метод контроля ошибки
            dists[i][0] = distance(learnData.loc[passengerIndexForPrediction,], row)
            dists[i][1] = row['Survived']
            i += 1
    dists = sorted(dists, key = lambda pair: pair[0])
    prediction = 0
    for i in range(K):
        prediction += dists[i][1]
    prediction /= K
    return round(prediction)


accuracy = 0
for idx, row in tqdm.tqdm(data.iterrows(), total=len(data)):
    accuracy += row['Survived'] == myKNeighborsClassifier(data, 20, idx)
print(accuracy/data.shape[0])

