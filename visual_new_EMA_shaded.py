import pandas as pd
import numpy as np
import wandb
import random
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def smooth(smoothing_weight, viewport_scale, x_values, y_values):
    # Initialize variables
    last_y = 0 if len(y_values) > 0 else np.nan
    debias_weight = 0
    ema_values = []

    # Calculate the range of x (if needed for scaling)
    range_of_x = x_values.max() - x_values.min()

    # Calculate EMA with variable intervals
    for index, y_point in enumerate(y_values):
        prev_x = x_values.iloc[index - 1] if index > 0 else x_values.iloc[0]
        change_in_x = ((x_values.iloc[index] - prev_x) / range_of_x) * viewport_scale
        smoothing_weight_adj = np.power(smoothing_weight, change_in_x)
        
        last_y = last_y * smoothing_weight_adj + y_point
        debias_weight = debias_weight * smoothing_weight_adj + 1
        ema_value = last_y / debias_weight
        ema_values.append(ema_value)
    
    return ema_values


if __name__ == "__main__":

    # Load your data
    FetchPush = pd.read_csv('/home/nikisim/Downloads/UR5_FetchPush_layer2.csv')

    # original_array = FetchPush['FetchPush_new_8 (Run set) - eval/is_succeess'].dropna().to_numpy()

    # # Create an array of indices for the original array
    # original_indices = np.linspace(0, 1, num=len(original_array))

    # # Create an array of indices for the new array with 1810 elements
    # new_indices = np.linspace(0, 1, num=1810)

    # # Use interpolation to create the new array
    # # 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic' are some of the options
    # interpolation_method = 'linear'  # Choose the method you prefer
    # interpolator = interp1d(original_indices, original_array, kind=interpolation_method)

    # orig_ddpg = interpolator(new_indices)


    # Extract the series you want to smooth
    x_rebrac_1 = FetchPush['Step']
    y_rebrac_4_4_1 = FetchPush['ReBRAC_4_4_1 (Run set) - eval/is_succeess']
    y_rebrac_4_4_2 = FetchPush['ReBRAC_4_4_2 (Run set) - eval/is_succeess']
    y_rebrac_4_4_3 = FetchPush['ReBRAC_4_4_3 (Run set) - eval/is_succeess']

    y_rebrac_3_3_1 = FetchPush['ReBRAC_3_3_1 (Run set) - eval/is_succeess']
    y_rebrac_3_3_2 = FetchPush['ReBRAC_3_3_2 (Run set) - eval/is_succeess']
    y_rebrac_3_3_3 = FetchPush['ReBRAC_3_3_3 (Run set) - eval/is_succeess']

    y_rebrac_3_4_1 = FetchPush['ReBRAC_3_4_1 (Run set) - eval/is_succeess']
    y_rebrac_3_4_2 = FetchPush['ReBRAC_3_4_2 (Run set 2) - eval/is_succeess']
    # y_rebrac_4 = FetchPush['rebrac-Unitree_ETG_Ground-e0d921a2 (Run set) - eval/return_mean']
    # y_rebrac_5 = FetchPush['rebrac-Unitree_ETG_Ground-2538adf8 (Run set) - eval/return_mean']
   
    # y_iql = FetchPush['IQL-FetchReach_UR5-270e756c (Run set 2) - eval/is_succeess']

    # y_iql  = y_iql.drop([0])

    dict1 = {
        'Step': x_rebrac_1,#.to_numpy()[:-1],
        'ReBRAC_4_4_1': y_rebrac_4_4_1,#.to_numpy()[:-1],
        'ReBRAC_4_4_2': y_rebrac_4_4_2,#.to_numpy()[:-1],
        'ReBRAC_4_4_3': y_rebrac_4_4_3,#.to_numpy()[:-1],

        'ReBRAC_3_3_1': y_rebrac_3_3_1,#.to_numpy()[:-1],
        'ReBRAC_3_3_2': y_rebrac_3_3_2,#.to_numpy()[:-1],
        'ReBRAC_3_3_3': y_rebrac_3_3_3,#.to_numpy()[:-1],

        'ReBRAC_3_4_1': y_rebrac_3_3_1,#.to_numpy()[:-1],
        'ReBRAC_3_4_2': y_rebrac_3_3_2,#.to_numpy()[:-1],
        # 'ReBRAC_4': y_rebrac_4,#.to_numpy()[:-1],
        # 'ReBRAC_5': y_rebrac_5,#.to_numpy()[:-1],
        # 'IQL': y_iql
    }

    df = pd.DataFrame(dict1).dropna()

    #adding zeros on the top
    # df.loc[0] = [0, 0.0, 0.0,0.0]
    # df.index = df.index + 1  # shifting index
    # df.sort_index(inplace=True) 

    # Define the smoothing parameter
    smoothing_param = 0.99  # You can adjust this value as needed
    smoothing_weight = min(np.sqrt(smoothing_param), 0.999)
    viewport_scale = 1  # Adjust this if you need to scale the result to a specific range

    smooth_rebrac_4_4_1 = smooth(smoothing_weight, viewport_scale, df['Step'], df['ReBRAC_4_4_1'])
    smooth_rebrac_4_4_2 = smooth(smoothing_weight, viewport_scale, df['Step'], df['ReBRAC_4_4_2'])
    smooth_rebrac_4_4_3 = smooth(smoothing_weight, viewport_scale, df['Step'], df['ReBRAC_4_4_3'])

    smooth_rebrac_3_3_1 = smooth(smoothing_weight, viewport_scale, df['Step'], df['ReBRAC_3_3_1'])
    smooth_rebrac_3_3_2 = smooth(smoothing_weight, viewport_scale, df['Step'], df['ReBRAC_3_3_2'])
    smooth_rebrac_3_3_3 = smooth(smoothing_weight, viewport_scale, df['Step'], df['ReBRAC_3_3_3'])

    smooth_rebrac_3_4_1 = smooth(smoothing_weight, viewport_scale, df['Step'], df['ReBRAC_3_4_1'])
    smooth_rebrac_3_4_2 = smooth(smoothing_weight, viewport_scale, df['Step'], df['ReBRAC_3_4_2'])
    # smooth_rebrac_4 = smooth(smoothing_weight, viewport_scale, df['Step'], df['ReBRAC_4'])
    # smooth_rebrac_5 = smooth(smoothing_weight, viewport_scale, df['Step'], df['ReBRAC_5'])
    # smooth_iql = smooth(smoothing_weight, viewport_scale, df['Step'], df['IQL'])
    # smooth_iql = smooth(smoothing_weight, viewport_scale, df['Step'], df['IQL'])
    # smooth_ddpg = smooth(smoothing_weight, viewport_scale, df['Step'], df['DDPG'])

    # # Calculate the range of x (if needed for scaling)
    # range_of_x = x_rebrac_1_1.max() - x_rebrac_1_1.min()

    # # Fit the model
    # model_rebrac = SimpleExpSmoothing(y_rebrac_1_1).fit(smoothing_level=0.05, optimized=False)
    # model_iql = SimpleExpSmoothing(y_iql_100[:leng]).fit(smoothing_level=0.05, optimized=False)

    # # Get the smoothed data
    # smoothed_rebrac = model_rebrac.fittedvalues
    # smoothed_iql = model_iql.fittedvalues

    experiment1 = np.array([smooth_rebrac_4_4_1,smooth_rebrac_4_4_2,smooth_rebrac_4_4_3])
    experiment2 = np.array([smooth_rebrac_3_3_1,smooth_rebrac_3_3_2,smooth_rebrac_3_3_3])
    experiment3 = np.array([smooth_rebrac_3_4_1,smooth_rebrac_3_4_2])
    # Example data for 3 runs of 2 experiments (each run has 10 points)
    # experiment1 = np.array([
    #     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #     [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1],
    #     [0.9, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9, 9.9]
    # ])
    # experiment2 = np.array([
    #     [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    #     [2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2, 11.2],
    #     [1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8, 8.8, 9.8, 10.8]
    # ])

    # Compute means and standard deviations
    mean1 = np.mean(experiment1, axis=0)
    std1 = np.std(experiment1, axis=0)

    mean2 = np.mean(experiment2, axis=0)
    std2 = np.std(experiment2, axis=0)

    mean3 = np.mean(experiment3, axis=0)
    std3 = np.std(experiment3, axis=0)
    

    plt.grid(linestyle='-')

    # Plot mean and standard deviation for experiment 1
    plt.plot(df['Step'], mean1, label='ReBRAC 4 4 слоя', color='blue')
    plt.fill_between(df['Step'], mean1 - std1, mean1 + std1, color='blue', alpha=0.2)

    # Plot mean and standard deviation for experiment 2
    plt.plot(df['Step'], mean2, label='ReBRAC 3 3 слоя', color='red')
    plt.fill_between(df['Step'], mean2 - std2, mean2 + std2, color='red', alpha=0.2)

    # Plot mean and standard deviation for experiment 2
    # plt.plot(df['Step'], mean3, label='ReBRAC_3_4_lay Mean', color='green')
    # plt.fill_between(df['Step'], mean3 - std3, mean3 + std3, color='green', alpha=0.2)


    # plt.plot(df['Step'],smooth_rebrac_1)
    # plt.plot(df['Step'],smooth_rebrac_2)
    # plt.plot(df['Step'],smooth_rebrac_3)
    # plt.plot(df['Step'],smooth_rebrac_4)
    # plt.plot(df['Step'],smooth_rebrac_5)
    # plt.plot(df['Step'],smooth_iql)
    # plt.plot(x_rebrac_10_10[:leng],smooth(y_rebrac_10_10.to_numpy(), radius=sm))
    # plt.plot(df['Step'],smooth_iql)
    # plt.plot(df['Step'],smooth_ddpg)

    plt.ylim(0.65,0.9)
    plt.xlim(0.7,3300)
    #plt.legend(['ReBRAC_4_4_lay','ReBRAC_3_3_lay'], loc=4)
    plt.legend()
    plt.title('Среда FetchPickAndPlace')
    plt.xlabel('Кол-во эпох')
    plt.ylabel('Доля успешных эпизодов')
    # plt.savefig('/home/nikisim/Mag_diplom/CORL/Images/FetchReach.png')
    plt.show()

    # data = [[x, y] for (x, y) in zip(df['Step'], smooth_rebrac_3)]

    # # # Start a new run
    # run = wandb.init(project='FetchPickAndPlace_inverse_compare_1', name='ReBRAC_0.1_0.1')

    # # # Create a table with the columns to plot
    # table = wandb.Table(data=data, columns=["Кол-во эпох", "Доля успешных эпизодов"])

    # # # Use the table to populate various custom charts
    # line_plot = wandb.plot.line(table, x='Кол-во эпох', y='Доля успешных эпизодов', title='Среда FetchPickAndPlace')

    # # # Log custom tables, which will show up in customizable charts in the UI
    # wandb.log({'line_1': line_plot, 
    #             })

    # # # Finally, end the run. We only need this ine in Jupyter notebooks.
    # run.finish()