import matplotlib.pyplot as plt
import  numpy as np
class MyPlot:
    def __init__(self):
        self.start = 0
        self.end = None
        self.pace = None
        self.data = None
        self.x_label = None
        self.y_label = None
        self.label = None
        self.plot()

    def __init__(self,end,pace,data):
        self.start = 0
        self.end = end
        self.pace = pace
        self.data = data
        self.label = None
        self.x_label = None
        self.y_label = None
        self.plot()

    def plot(self,show=True):
        if self.end is None or \
                self.pace is None or \
                self.data is None or\
                not show:
            if show:
                print('Icomplete data for plotting signal')
            return None
        x = np.arange(self.start, self.end, self.pace)
        if  self.label is not None:
            plt.plot(x,self.data,label=self.label)
        else:
            plt.plot(x,self.data)
        if self.x_label is not None:
            plt.xlabel(self.x_label)
        if self.y_label is not None:
            plt.ylabel(self.y_label)
