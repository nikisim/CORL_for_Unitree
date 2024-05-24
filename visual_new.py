import pandas as pd
import numpy as np
import wandb
import random
import math
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing

def smooth(y, radius, mode='two_sided', valid_only=False):
    '''
    Smooth signal y, where radius is determines the size of the window

    mode='twosided':
        average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode='causal':
        average over the window [max(index - radius, 0), index]

    valid_only: put nan in entries where the full-sized window is not available

    '''
    assert mode in ('two_sided', 'causal')
    if y.shape[0] < 2*radius+1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius+1)
        out = np.convolve(y, convkernel,mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel,mode='full') / np.convolve(np.ones_like(y), convkernel, mode='full')
        out = out[:-radius+1]
        if valid_only:
            out[:radius] = np.nan
    return out


if __name__ == "__main__":

    # Load your data
    FetchPush = pd.read_csv('/home/nikisim/Downloads/UR5_FetchReach.csv')

    # Extract the series you want to smooth
    x_rebrac_1_1 = FetchPush['Step']
    y_rebrac_1_1 = FetchPush['ReBRAC_400.0_400.0--48bf5f6b (Run set 2) - eval/is_succeess'].dropna()

    x_iql_100 = FetchPush['Step']
    y_iql_100 = FetchPush['IQL-FetchReach_UR5-270e756c (Run set 2) - eval/is_succeess'].dropna()

    leng = y_rebrac_1_1.shape[0]

    # Fit the model
    model_rebrac = SimpleExpSmoothing(y_rebrac_1_1).fit(smoothing_level=0.05, optimized=False)
    model_iql = SimpleExpSmoothing(y_iql_100[:leng]).fit(smoothing_level=0.05, optimized=False)

    # Get the smoothed data
    smoothed_rebrac = model_rebrac.fittedvalues
    smoothed_iql = model_iql.fittedvalues

    plt.grid(linestyle='-')
    plt.plot(x_rebrac_1_1[:leng],smoothed_rebrac)
    # plt.plot(x_rebrac_10_10[:leng],smooth(y_rebrac_10_10.to_numpy(), radius=sm))
    plt.plot(x_iql_100[:leng],smoothed_iql)

    # plt.ylim(0.75,1.01)
    # plt.xlim(0.7,601.5)
    plt.legend(['ReBRAC_1_1','IQL_100'], loc=4)
    plt.title('Среда FetchPush')
    plt.xlabel('Кол-во эпох')
    plt.ylabel('Доля успешных эпизодов')
    # plt.savefig('/home/nikisim/Mag_diplom/CORL/Images/FetchReach.png')
    plt.show()

    # FetchPush = pd.read_csv('/home/nikisim/Downloads/Telegram Desktop/FetchPush.csv')

    # x_rebrac_1_1 = FetchPush['Step']
    # y_rebrac_1_1 = FetchPush['rebrac-FetchPushDense-v2-b3d3046c (Run set) - eval/is_succeess'].dropna()

    # x_iql_100 = FetchPush['Step']
    # y_iql_100 = FetchPush['IQL-FetchPush-3918fdbe (Run set 2) - eval/is_succeess'].dropna()

    # leng = y_rebrac_1_1.shape[0]

    # data = [[x, y] for (x, y) in zip(y_iql_100[:leng].to_numpy(), smooth(x_iql_100.to_numpy(), radius=99))]

    # # Start a new run
    # run = wandb.init(project='FetchPush_vis_test')

    # # Create a table with the columns to plot
    # table = wandb.Table(data=data, columns=["epoch", "success_rate"])

    # # Use the table to populate various custom charts
    # line_plot = wandb.plot.line(table, x='epoch', y='success_rate', title='FetchReach')

    # # Log custom tables, which will show up in customizable charts in the UI
    # wandb.log({'line_1': line_plot, 
    #             })

    # # Finally, end the run. We only need this ine in Jupyter notebooks.
    # run.finish()