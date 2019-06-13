import os
import os.path as osp
import csv
import sys
import matplotlib.pyplot as plt
import pandas as pd
from baselines import logger

def createPlotPath():
    plotdir = osp.join(logger.get_dir(), 'plots')
    os.makedirs(plotdir, exist_ok=True)
    return plotdir

def createCSV(plotdir, num):
    with open(('%s/ep%.5i.csv' % (plotdir, num)), 'w', newline='') as csvfile:
        fieldnames = ['timesteps', 'xpos', 'ypos']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    print('Episode %i CSV created at %s' % (num, plotdir))

def newRow(plotdir, num, timesteps, xpos, ypos):
    with open(('%s/ep%.5i.csv' % (plotdir,num)), 'a', newline='') as csvfile:
        fieldnames = ['timesteps', 'xpos', 'ypos']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'timesteps':timesteps, 'xpos':xpos, 'ypos':ypos})

def genPlot(plotdir, num):
    df = pd.read_csv('%s/ep%.5i.csv' % (plotdir,num))
    plt.plot('timesteps', 'xpos', data=df, marker='', color='red', linewidth=2)
    plt.plot('timesteps', 'ypos', data=df, marker='', color='blue', linewidth=2)
    plt.title('Episode %i' % num)
    plt.xlabel('Timesteps')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
#
# def main():
#    genPlot('C:/Users/Admin/AppData/Local/Temp/openai-2019-06-13-15-48-45-318250/plots', 1)
#
#if __name__ == '__main__':
#    main()