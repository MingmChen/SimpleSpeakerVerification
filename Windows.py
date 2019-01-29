import numpy as np
from scipy import cos,pi,log10
from scipy.special import iv
class HannWindow:
    def __init__(self,M):
        """
        Create hanning window:

        w[n] = 0.5-0.5*cos(2*pi*n/M)

        :param M: Length of window
        """
        self.window = np.arange(M)
        # print(self.window)
        self.window = 0.5-0.5*cos(2*pi*self.window/M)


class HammingWindow:
    def __init__(self,M):
        """
        Create Hamming window

        w[n] = 0.54-0.46*cos(2*pi*n/M)
        :param M: Length of window
        """
        self.window = np.arange(M)
        self.window = 0.54 - 0.46*cos(2*pi*self.window/M)


class KaiserWindow:

    def computeBetha(self):
        if  self.A < 21:
            self.Betha  = 0
        elif self.A <=50:
            self.Betha = 0.5842 * np.power((self.A -21),0.4) + 0.07886 * (self.A-21)
        else:
            self.Betha = 0.1102(self.A-8.7)

    def computeM(self,deltaW):
        self.M = (self.A-8)/(2.285*deltaW)
    def __init__(self, deltaW, delta):
        self.computeM(deltaW)
        alpha = self.M /2
        self.A = -20 * log10(delta)
        self.computeBetha(deltaW)
        self.window = iv(0, self.Betha*np.sqrt(1 - np.power((np.arange(self.M)-alpha)/alpha, 2))) / iv(0, self.Betha)
    def __init__(self,M,Betha):
        self.Betha = Betha
        alpha = M/2
        self.window = iv(0, self.Betha*np.sqrt(1 - np.power((np.arange(M)-alpha)/alpha, 2))) / iv(0, self.Betha)

