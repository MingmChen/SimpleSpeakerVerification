
import  numpy as np
from config import FB_UPPER_FREQUENCY,FB_LOWER_FREQUENCY,FB_FILTER_NUM,FFTNUM, FB_TYPE
def hzToMel(f):
    return 1125 * np.log(1+f/700)
def melToHz(mel):
    return 700 * (np.exp(mel/1125)-1)

def filterBank(sampleRate):
    """
    Return filter bank.
    Uses variables from config:
        FB_UPPER_FREQUENCY : upper frequency in Hz
        FB_LOWER_FREQUENCY : lower frequency in Hz
        FB_TYPE            : window type i.e. triangular , etc
    :param Hz: If true returning result  will be in Hz.If false returning result will be in Mel. scale
    :return:An array of size FB_FILTER_NUM+2
    """
    upperMel = hzToMel(FB_UPPER_FREQUENCY)
    lowerMel = hzToMel(FB_LOWER_FREQUENCY)
    result = np.linspace(lowerMel, upperMel,FB_FILTER_NUM+2)
    result = melToHz(result)
    final = np.floor((result * (FFTNUM+1)/sampleRate))


    filterBank = np.zeros((FB_FILTER_NUM,int(np.floor(FFTNUM/2 ))))
    if FB_TYPE == 'triangular':
        print('Triangular filter bank.')
        for i in range(1,FB_FILTER_NUM+1):
            left = int(final[i-1])
            center = int(final[i])
            right = int(final[i+1])

            for k in range(left,center):
                filterBank[i-1,k] = (k-final[i-1])/(final[i]-final[i-1])
            for k in range(center,right):
                filterBank[i-1,k] = (final[i+1]-k)/(final[i+1]-final[i])
    return filterBank
