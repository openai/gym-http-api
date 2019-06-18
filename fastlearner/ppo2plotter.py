import os
import os.path as osp
import csv
import sys
import matplotlib.pyplot as plt
import pandas as pd
from baselines import logger

class Plotter(object):

    def __init__(self, epnum, eptime, start_x, start_y, last_x, last_y, dead):
        self.epnum = epnum
        self.eptime = eptime
        self.start_x = start_x
        self.start_y = start_y
        self.last_x = last_x
        self.last_y = last_y
        self.dead = dead

    def getEpNum(self):
        return self.epnum

    def getEpTime(self):
        return self.eptime

    def updateEpNum(self):
        self.epnum += 1
        return self.epnum

    def updateEpTime(self):
        self.eptime += 1
        return self.eptime

    def resetEpTime(self):
        self.eptime = 0
        return self.eptime

    def passOnStart(self, x, y):
        self.start_x = x
        self.start_y = y
        
    def passOnLast(self, x, y):
        self.last_x = x
        self.last_y = y

    def getStartX(self):
        return self.start_x

    def getStartY(self):
        return self.start_y

    def getLastX(self):
        return self.last_x

    def getLastY(self):
        return self.last_y

    def setDeath(self, num):
        self.dead = num

    def isDead(self):
        return self.dead
        

def createPlotPath():
    plotdir = osp.join(logger.get_dir(), 'plots')
    os.makedirs(plotdir, exist_ok=True)
    return plotdir

def createCSV(plotdir, num):
    with open(('%s/data%.5i.csv' % (plotdir, num)), 'w', newline='') as csvfile:
        fieldnames = ['timesteps', 'xpos', 'ypos']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    print('Episode %i CSV created at %s' % (num, plotdir))

def newRow(plotdir, num, timesteps, xpos, ypos):
    with open(('%s/data%.5i.csv' % (plotdir,num)), 'a', newline='') as csvfile:
        fieldnames = ['timesteps', 'xpos', 'ypos']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'timesteps':timesteps, 'xpos':xpos, 'ypos':ypos})

def genPlot(plotdir, num, x_lim):
    df = pd.read_csv('%s/data%.5i.csv' % (plotdir,num))
    plt.xlim(0, x_lim)
    plt.ylim(0, 1500)
    plt.plot('xpos', 'ypos', data=df, marker='', color='#202088', linewidth=2)
    plt.title('Episode %i' % num)
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.savefig('%s/graph%.5i.png' % (plotdir,num))
    plt.close()
    print('Episode %i plot created at %s' % (num, plotdir))

def main():
    genPlot('C:/Users/Admin/AppData/Local/Temp/openai-2019-06-14-14-15-07-702276/plots', 1)

if __name__ == '__main__':
    main()