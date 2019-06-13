import os
import os.path as osp
import csv
import matplotlib.pyplot as plt
import pandas as pd
from baselines import logger

def createPlotPath():
    plotdir = osp.join(logger.get_dir(), 'plots')
    os.makedirs(plotdir, exist_ok=True)
    print(plotdir)

def createCSV(num):
    with open(('%s/data%i.csv' % (plotdir,num)), 'w', newline='') as csvfile:
        fieldnames = ['timesteps', 'xpos', 'ypos']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    return writer

def newRow(writer, timesteps, xpos, ypos): writer.writerow({'timesteps':timesteps, 'xpos':xpos, 'ypos':ypos})

def genPlot(num):
    df = pd.read_csv('%s/data%i.csv' % (plotdir,num))
    plt.plot('timesteps', 'xpos', data=df, marker='', color='red', linewidth=2)
    plt.plot('timesteps', 'ypos', data=df, marker='', color='blue', linewidth=2)
    plt.title('Episode %i' % num)
    plt.xlabel('Timesteps')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    

def main():
    for num in range(5):
        createCSV(num + 1)
    genPlot(2)

if __name__ == '__main__':
    main()

#if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():

#plotdir = osp.join(logger.get_dir(), 'plots')
#            os.makedirs(plotdir, exist_ok=True)
#            savepath = osp.join

# writer = createCSV(update)
# newRow(writer
# 
