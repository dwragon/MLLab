import np as np
import pandas as pd
from sklearn.neighbors import KernelDensity

import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")

#вариант №7 ядро №2
'''i0 = data['Response'] == 0
kde0 = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(data.loc[i0, 'L1_S24_F1846'].values.reshape(-1, 1))
X_plot = np.linspace(-1, 1, 1000).reshape(-1, 1)
Dens0 = np.exp(kde0.score_samples(X_plot))  # score_samples возвращает логарифм плотности

i1 = data['Response'] == 1
kde1 = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(data.loc[i1, 'L1_S24_F1846'].values.reshape(-1, 1))
Dens1 = np.exp(kde1.score_samples(X_plot))


plt.plot(X_plot, Dens0, '.', color='red')
plt.plot(X_plot, Dens1, '.', color='blue')
plt.show()'''
#выборка хорошо разделима

dataTrain = data.loc[0:data.shape[0] / 2, ].reset_index(drop=True)
dataTest = data.loc[data.shape[0] / 2:data.shape[0], ].reset_index(drop=True)

r = 0
kde0 = KernelDensity(kernel='gaussian', bandwidth=0.05)
kde0.fit(dataTrain.loc[dataTrain['Response']==r, 'L1_S24_F1846'].values.reshape(-1, 1))
logProbability0 = kde0.score_samples(dataTest.loc[dataTest['Response']==r, 'L1_S24_F1846'].values.reshape(-1, 1))
logProbability0[np.isinf(logProbability0)] = -100 # заменяем -бесконечность
logLikehood0 = logProbability0.sum()
#print(logLikehood0)

r = 1
kde1 = KernelDensity(kernel='gaussian', bandwidth=0.05)
kde1.fit(dataTrain.loc[dataTrain['Response']==r, 'L1_S24_F1846'].values.reshape(-1, 1))
logProbability1 = kde1.score_samples(dataTest.loc[dataTest['Response']==r, 'L1_S24_F1846'].values.reshape(-1, 1))
logProbability1[np.isinf(logProbability1)] = -100 # заменяем -бесконечность
logLikehood1 = logProbability1.sum()
#print(logLikehood1)


predictionProbXafter0 = np.exp(kde0.score_samples(dataTest['L1_S24_F1846'].values.reshape(-1, 1)))
predictionProbXafter1 = np.exp(kde1.score_samples(dataTest['L1_S24_F1846'].values.reshape(-1, 1)))

predictionProb0afterX =  predictionProbXafter0 #тут должна быть формула Баеса
predictionProb1afterX = predictionProbXafter1 #тут должна быть формула Баеса

ind = np.argsort(predictionProb0afterX)# сортировка, возвращающая индексы элементов
print(predictionProb0afterX[ind[-10:]]) # вывод последних 10 элементов
print(sum(dataTest.loc[ind[-100:],'Response'])) # количество бракованных среди 100 с максимальной вероятностью брака
ind1 = np.argsort(predictionProb1afterX)# сортировка, возвращающая индексы элементов
print(predictionProb1afterX[ind1[-10:]]) # вывод последних 10 элементов
print(sum(dataTest.loc[ind1[-100:],'Response'])) # количество бракованных среди 100 с максимальной вероятностью брака








#вариант №11 ядро №6
'''i0 = data['Response'] == 0
kde0 = KernelDensity(kernel='exponential', bandwidth=0.1).fit(data.loc[i0, 'L1_S24_F1846'].values.reshape(-1, 1))
X_plot = np.linspace(-1, 1, 1000).reshape(-1, 1)
Dens0 = np.exp(kde0.score_samples(X_plot))  # score_samples возвращает логарифм плотности

i1 = data['Response'] == 1
kde1 = KernelDensity(kernel='exponential', bandwidth=0.1).fit(data.loc[i1, 'L1_S24_F1846'].values.reshape(-1, 1))
Dens1 = np.exp(kde1.score_samples(X_plot))


plt.plot(X_plot, Dens0, '.', color='red')
plt.plot(X_plot, Dens1, '.', color='blue')
plt.show()'''
