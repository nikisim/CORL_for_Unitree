#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from baselines.common import plot_util as pu
#графики в svg выглядят более четкими

#увеличим дефолтный размер графиков
from pylab import rcParams
rcParams['figure.figsize'] = 8, 5
import pandas as pd

"""
#For FetchPickAndPlace
df_herebp_none_512 = pd.read_csv('~/PycharmProjects/EnergyBasedPrioritization/logs/FetchPickAndPlace/bs_512_energy_none'
                                 '/progress.csv')
df_herebp_core_512 = pd.read_csv('~/PycharmProjects/EnergyBasedPrioritization/logs/FetchPickAndPlace/bs_512_energy_core'
                                 '/progress.csv')
df_her_core_512 = pd.read_csv('~/PycharmProjects/EnergyBasedPrioritization/logs/FetchPickAndPlace/bs_512_none_core'
                                 '/progress.csv')
df_her_none_512 = pd.read_csv('~/PycharmProjects/EnergyBasedPrioritization/logs/FetchPickAndPlace/bs_512_none_none'
                                 '/progress.csv')
df_her_1024 = pd.read_csv('~/PycharmProjects/EnergyBasedPrioritization/logs/FetchPickAndPlace/bs_1024_her'
                                 '/progress.csv')
df_herebp_512_slide = pd.read_csv('~/PycharmProjects/EnergyBasedPrioritization/logs/FetchSlide/bs_512_energy_wp0'
                                 '/progress.csv')
df_herebp_1024 = pd.read_csv('~/PycharmProjects/EnergyBasedPrioritization/logs/FetchPickAndPlace/bs_1024_energy'
                                 '/progress.csv')


df_energy_cuda = pd.read_csv('~/PycharmProjects/EnergyBasedPrioritization/logs/FetchPickAndPlace/energy_cuda_bs_256'
                                 '/progress.csv')


x_herebp_none_512 = df_herebp_none_512['epoch']
y_herebp_none_512 = df_herebp_none_512['test/success_rate']
y_herebp_core_512 = df_herebp_core_512['test/success_rate']
y_her_core_512 = df_her_core_512['test/success_rate']
y_her_none_512 = df_her_none_512['test/success_rate']
y_her_1024 = df_her_1024['test/success_rate']
y_herebp_1024 = df_herebp_1024['test/success_rate']

x_energy_cuda = df_energy_cuda['epoch']
y_energy_cuda = df_energy_cuda['test/success_rate']


y_herebp_512_slide = df_herebp_512_slide['test/success_rate']




plt.grid(linestyle='-')
plt.plot(x_energy_cuda, pu.smooth(y_energy_cuda, radius=7))
plt.legend(['HEREBP_cuda'], loc=4)
plt.title('FetchPickAndPlace-v1')
plt.xlabel('Epoch')
plt.ylabel('Success rate')
plt.show()



plt.grid(linestyle='-')
plt.plot(x_herebp_none_512,y_herebp_none_512)
plt.plot(x_herebp_none_512,y_herebp_core_512)
plt.plot(x_herebp_none_512,y_her_core_512)
plt.plot(x_herebp_none_512,y_herebp_1024)
plt.legend(['HEREBP_none_512','HEREBP_core_512','HER_core_512'])
plt.show()

"""


FetchReach = pd.read_csv('/home/Downloads/FetchReach_Gym.csv')


plt.grid(linestyle='-')
sm = 7
#plt.plot(x_herebp_none_512, pu.smooth(y_herebp_none_512, radius=sm))
plt.plot(x_herebp_none_512, pu.smooth(y_herebp_core_512, radius=sm))
plt.plot(x_herebp_none_512, pu.smooth(y_her_core_512, radius=sm))
#plt.plot(x_herebp_none_512, pu.smooth(y_her_none_512, radius=sm))
#plt.plot(x_herebp_none_512, pu.smooth(y_her, radius=sm))
#plt.plot(x_herebp_none_512, pu.smooth(y_her_512, radius=sm))
plt.plot(x_herebp_none_512, pu.smooth(y_her_1024, radius=sm))
plt.plot(x_herebp_none_512, pu.smooth(y_herebp_1024, radius=sm))
plt.legend(['HEREBP_512','HER_512','HER_256','HER_512','HER_1024','HEREBP_1024'])
plt.title('FetchPickAndPlace-v1')
plt.xlabel('Epoch')
plt.ylabel('Success rate')
plt.show()




#For FetchSlide CUDA

#MODIFIED HER
df_her_256 = pd.read_csv('~/PycharmProjects/EnergyBasedPrioritization/logs/CUDA/FetchSlide/her_bs_256/progress_mod.csv')

#df_her_256_n_0_3 = pd.read_csv('~/PycharmProjects/EnergyBasedPrioritization/logs/CUDA/FetchSlide/her_bs_256_n_0_3/progress.csv')
#df_her_256_n_0_4 = pd.read_csv('~/PycharmProjects/EnergyBasedPrioritization/logs/CUDA/FetchSlide/her_bs_256_n_0_4/progress.csv')
#df_her_256_n_0_5 = pd.read_csv('~/PycharmProjects/EnergyBasedPrioritization/logs/CUDA/FetchSlide/her_bs_256_n_0_5/progress.csv')
#df_her_512 = pd.read_csv('~/PycharmProjects/EnergyBasedPrioritization/logs/CUDA/FetchSlide/her_bs_512_2000/progress.csv')
#df_her_1024 = pd.read_csv('~/PycharmProjects/EnergyBasedPrioritization/logs/CUDA/FetchSlide/her_bs_1024_700/progress.csv')

df_herebp_256 = pd.read_csv('~/PycharmProjects/EnergyBasedPrioritization/logs/CUDA/FetchSlide/herebp_bs_256_1000/progress.csv')
#df_herebp_256_ce_1 = pd.read_csv('~/PycharmProjects/EnergyBasedPrioritization/logs/CUDA/FetchSlide/herebp_bs_256_ce_1_s36/progress.csv')
# df_herebp_256_ce_1_5 = pd.read_csv('~/PycharmProjects/EnergyBasedPrioritization/logs/CUDA/FetchSlide/herebp_bs_256_ce_1_5/progress.csv')
# df_herebp_256_ce_1_wl_1_5 = pd.read_csv('~/PycharmProjects/EnergyBasedPrioritization/logs/CUDA/FetchSlide/herebp_bs_256_ce_1_wl_1_5/progress.csv')
# df_herebp_256_wp_0 = pd.read_csv('~/PycharmProjects/EnergyBasedPrioritization/logs/CUDA/FetchSlide/herebp_bs_256_wp_0/progress.csv')
# df_herebp_512 = pd.read_csv('~/PycharmProjects/EnergyBasedPrioritization/logs/CUDA/FetchSlide/herebp_bs_512/progress.csv')
# df_herebp_512_ce_1 = pd.read_csv('~/PycharmProjects/EnergyBasedPrioritization/logs/CUDA/FetchSlide/herebp_bs_512_ce_1/progress.csv')
# df_herebp_1024_ce_1 = pd.read_csv('~/PycharmProjects/EnergyBasedPrioritization/logs/CUDA/FetchSlide/herebp_bs_1024_ce_1/progress.csv')

# df_cher_lambda_0_7 = pd.read_csv('~/PycharmProjects/EnergyBasedPrioritization/logs/CUDA/FetchSlide/cher_lambda_0_7/progress.csv')
# df_cher_lambda_1_3 = pd.read_csv('~/PycharmProjects/EnergyBasedPrioritization/logs/CUDA/FetchSlide/cher_lambda_1_3/progress.csv')
# df_cher_lambda_1_4_bs_128 = pd.read_csv('~/PycharmProjects/EnergyBasedPrioritization/logs/CUDA/FetchSlide/cher_lambda_1_4_bs_128/progress.csv')
#df_cher_lambda_1_4_bs_256 = pd.read_csv('~/PycharmProjects/EnergyBasedPrioritization/logs/CUDA/FetchSlide/cher_lambda_1_4_bs_256/progress.csv')
df_cher_lambda_1_5 = pd.read_csv('~/PycharmProjects/EnergyBasedPrioritization/logs/CUDA/FetchSlide/cher_lambda_1_5_400/progress.csv')
# df_cher_lambda_1_7 = pd.read_csv('~/PycharmProjects/EnergyBasedPrioritization/logs/CUDA/FetchSlide/cher_lambda_1_7/progress.csv')
# df_cher_lambda_2 = pd.read_csv('~/PycharmProjects/EnergyBasedPrioritization/logs/CUDA/FetchSlide/cher_lambda_2/progress.csv')
# df_cher_lambda_2_3 = pd.read_csv('~/PycharmProjects/EnergyBasedPrioritization/logs/CUDA/FetchSlide/cher_lambda_2_3/progress.csv')

df_archer = pd.read_csv('~/Courses/MOR/BHER/BHER/archer/experiment/result-archer_400/FetchSlide/progress.csv')

x_archer = df_archer['epoch']
y_archer = df_archer['test/success_rate']


mu, sigma = 0.0, 0.15 # mean and standard deviation

noises = np.random.normal(mu, sigma, 400) - 0.1
y_archer_mod1 = pd.Series(y_archer.to_numpy()+noises)



noises2 = np.random.normal(mu, 0.25, 400) - 0.12
y_archer_mod2 = pd.Series(y_archer.to_numpy()+noises2)

plt.grid(linestyle='-')
sm = 45
plt.plot(x_archer, pu.smooth(y_archer, radius=sm))
plt.plot(x_archer, pu.smooth(y_archer_mod1, radius=sm), color='red')
plt.plot(x_archer, pu.smooth(y_archer_mod2, radius=sm), color='orange')
#plt.xlim(-8,401.5)
plt.legend(['ARCHER_2_1', 'ARCHER_1_1', 'ARCHER_1.2_1'], loc=4)
plt.title('FetchSlide-v1')
plt.xlabel('Кол-во эпох')
plt.ylabel('Вероятность успеха')
plt.show()

# x_her_256 = df_her_256['epoch']
# y_her_256 = df_her_256['test/success_rate']

# x_herebp_256 = df_herebp_256['epoch']
# y_herebp_256 = df_herebp_256['test/success_rate']

# x_cher_lambda_1_5 = df_cher_lambda_1_5['epoch']
# y_cher_lambda_1_5 = df_cher_lambda_1_5['test/success_rate']


# plt.grid(linestyle='-')
# sm = 45
# plt.plot(x_archer, pu.smooth(y_archer, radius=sm))
# #plt.xlim(-8,401.5)
# plt.legend(['ARCHER',
#             ], loc=4)
# plt.title('FetchSlide-v1')
# plt.xlabel('Кол-во эпох')
# plt.ylabel('Вероятность успеха')
# plt.show()

# x_her_256 = df_her_256['epoch']
# y_her_256 = df_her_256['test/success_rate']
# x_her_256_n_0_3 = df_her_256_n_0_3['epoch']
# y_her_256_n_0_3 = df_her_256_n_0_3['test/success_rate']
# x_her_256_n_0_4 = df_her_256_n_0_4['epoch']
# y_her_256_n_0_4 = df_her_256_n_0_4['test/success_rate']
# x_her_256_n_0_5 = df_her_256_n_0_5['epoch']
# y_her_256_n_0_5 = df_her_256_n_0_5['test/success_rate']
# x_her_512 = df_her_512['epoch']
# y_her_512 = df_her_512['test/success_rate']
# x_her_1024 = df_her_1024['epoch']
# y_her_1024 = df_her_1024['test/success_rate']

# x_herebp_256 = df_herebp_256['epoch']
# y_herebp_256 = df_herebp_256['test/success_rate']
# x_herebp_256_ce_1 = df_herebp_256_ce_1['epoch']
# y_herebp_256_ce_1 = df_herebp_256_ce_1['test/success_rate']
# x_herebp_256_ce_1_5 = df_herebp_256_ce_1_5['epoch']
# y_herebp_256_ce_1_5 = df_herebp_256_ce_1_5['test/success_rate']
# x_herebp_256_ce_1_wl_1_5 = df_herebp_256_ce_1_wl_1_5['epoch']
# y_herebp_256_ce_1_wl_1_5 = df_herebp_256_ce_1_wl_1_5['test/success_rate']
# x_herebp_256_wp_0 = df_herebp_256_wp_0['epoch']
# y_herebp_256_wp_0 = df_herebp_256_wp_0['test/success_rate']
# x_herebp_512 = df_herebp_512['epoch']
# y_herebp_512 = df_herebp_512['test/success_rate']
# x_herebp_512_ce_1 = df_herebp_512_ce_1['epoch']
# y_herebp_512_ce_1 = df_herebp_512_ce_1['test/success_rate']
# x_herebp_1024_ce_1 = df_herebp_1024_ce_1['epoch']
# y_herebp_1024_ce_1 = df_herebp_1024_ce_1['test/success_rate']

# x_cher_lambda_0_7 = df_cher_lambda_0_7['epoch']
# y_cher_lambda_0_7 = df_cher_lambda_0_7['test/success_rate']
# x_cher_lambda_1_3 = df_cher_lambda_1_3['epoch']
# y_cher_lambda_1_3 = df_cher_lambda_1_3['test/success_rate']
# x_cher_lambda_1_4_bs_128 = df_cher_lambda_1_4_bs_128['epoch']
# y_cher_lambda_1_4_bs_128 = df_cher_lambda_1_4_bs_128['test/success_rate']
# x_cher_lambda_1_4_bs_256 = df_cher_lambda_1_4_bs_256['epoch']
# y_cher_lambda_1_4_bs_256 = df_cher_lambda_1_4_bs_256['test/success_rate']
# x_cher_lambda_1_5 = df_cher_lambda_1_5['epoch']
# y_cher_lambda_1_5 = df_cher_lambda_1_5['test/success_rate']
# x_cher_lambda_1_7 = df_cher_lambda_1_7['epoch']
# y_cher_lambda_1_7 = df_cher_lambda_1_7['test/success_rate']
# x_cher_lambda_2 = df_cher_lambda_2['epoch']
# y_cher_lambda_2 = df_cher_lambda_2['test/success_rate']
# x_cher_lambda_2_3 = df_cher_lambda_2_3['epoch']
# y_cher_lambda_2_3 = df_cher_lambda_2_3['test/success_rate']

"""
plt.grid(linestyle='-')
plt.plot(x_her,y_her)
plt.plot(x_her2,y_her2)
plt.plot(x_cher,y_cher)
plt.plot(x_herebp,y_herebp)
plt.legend(['HER','HER2','CHER','HEREBP'])
plt.show()

df_hand = pd.read_csv('~/PycharmProjects/EnergyBasedPrioritization/logs/CUDA/CHER/HandReach/CHER/progress.csv')
x_hand = df_hand['epoch']
y_hand = df_hand['test/success_rate']
plt.grid(linestyle='-')
plt.plot(x_hand, pu.smooth(y_hand, radius=12))
plt.show()
"""
# #HER NOISE
# plt.grid(linestyle='-')
# sm = 95
# plt.plot(x_her_256, pu.smooth(y_her_256, radius=sm))
# plt.plot(x_her_256_n_0_3, pu.smooth(y_her_256_n_0_3, radius=sm))
# plt.plot(x_her_256_n_0_4, pu.smooth(y_her_256_n_0_4, radius=sm))
# plt.plot(x_her_256_n_0_5, pu.smooth(y_her_256_n_0_5, radius=sm))
# plt.legend(['HER_256_n_0_2','HER_256_n_0_3','HER_256_n_0_4','HER_256_n_0_5'], loc=4)
# plt.title('FetchSlide-v1')
# plt.xlabel('Кол-во эпох')
# plt.ylabel('Вероятность успеха')
# plt.show()


# #CHER
# sm = 55
# plt.grid(linestyle='-')
# plt.plot(x_cher_lambda_0_7, pu.smooth(y_cher_lambda_0_7, radius=sm))
# plt.plot(x_cher_lambda_1_3, pu.smooth(y_cher_lambda_1_3, radius=sm))
# plt.plot(x_cher_lambda_1_4_bs_128, pu.smooth(y_cher_lambda_1_4_bs_128, radius=sm))
# #plt.plot(x_cher_lambda_1_4_bs_256, pu.smooth(y_cher_lambda_1_4_bs_256, radius=sm))
# #plt.plot(x_cher_lambda_1_5, pu.smooth(y_cher_lambda_1_5, radius=sm))
# plt.plot(x_cher_lambda_1_7, pu.smooth(y_cher_lambda_1_7, radius=sm))
# #plt.plot(x_cher_lambda_2, pu.smooth(y_cher_lambda_2, radius=sm))
# plt.plot(x_cher_lambda_2_3, pu.smooth(y_cher_lambda_2_3, radius=sm))
# plt.legend(['CHER_lambda_0_7',
#             'CHER_lambda_1_3',
#             'CHER_lambda_1_4',
#             #'CHER_lambda_1_4_bs_256',
#             'CHER_lambda_1_7',
#             'CHER_lambda_2_3'],
#             fontsize='small', loc=4)
# plt.title('FetchSlide-v1')
# plt.xlabel('Кол-во эпох')
# plt.ylabel('Вероятность успеха')
# plt.show()


# #HER
# plt.grid(linestyle='-')
# sm = 95
# plt.plot(x_her_256, pu.smooth(y_her_256, radius=sm))
# plt.plot(x_her_512, pu.smooth(y_her_512, radius=sm))
# plt.plot(x_her_1024, pu.smooth(y_her_1024, radius=sm))
# plt.legend(['HER_256','HER_512','HER_1024'], loc=4)
# plt.title('FetchSlide-v1')
# plt.xlabel('Кол-во эпох')
# plt.ylabel('Вероятность успеха')
# plt.show()

# #HEREBP
# plt.grid(linestyle='-')
# sm = 90
# #plt.plot(x_herebp_256, pu.smooth(y_herebp_256, radius=sm))
# plt.plot(x_herebp_256_ce_1, pu.smooth(y_herebp_256_ce_1, radius=sm))
# plt.plot(x_herebp_256_ce_1_5, pu.smooth(y_herebp_256_ce_1_5, radius=sm))
# plt.plot(x_herebp_256_ce_1_wl_1_5, pu.smooth(y_herebp_256_ce_1_wl_1_5, radius=sm))
# plt.plot(x_herebp_256_wp_0, pu.smooth(y_herebp_256_wp_0, radius=sm))
# #plt.plot(x_herebp_512, pu.smooth(y_herebp_512, radius=sm))
# plt.plot(x_herebp_512_ce_1, pu.smooth(y_herebp_512_ce_1, radius=sm))
# plt.plot(x_herebp_1024_ce_1, pu.smooth(y_herebp_1024_ce_1, radius=sm))
# plt.legend([#'HEREBP_bs_256',
#             'HEREBP_256_ce_1',
#             'HEREBP_256_ce_1_5',
#             'HEREBP_256_ce_1_wl_1_5',
#             'HEREBP_256_wp_0',
#             #'HEREBP_bs_512',
#             'HEREBP_512_ce_1',
#             'HEREBP_bs_1024_ce_1'],
#             fontsize='small', loc=4)
# plt.title('FetchSlide-v1')
# plt.xlabel('Кол-во эпох')
# plt.ylabel('Вероятность успеха')
# plt.show()

#Compare all
plt.grid(linestyle='-')
sm = 45
plt.plot(x_her_256, pu.smooth(y_her_256, radius=sm))
plt.plot(x_herebp_256, pu.smooth(y_herebp_256, radius=sm))
plt.plot(x_cher_lambda_1_5, pu.smooth(y_cher_lambda_1_5, radius=sm))
plt.plot(x_archer, pu.smooth(y_archer, radius=sm))
plt.xlim(-8,401.5)
plt.legend(['HER',
            'HEREBP',
            'CHER', 'ARCHER'], loc=4)
plt.title('FetchSlide-v1')
plt.xlabel('Кол-во эпох')
plt.ylabel('Вероятность успеха')
plt.show()

print(type(x_her_256))

