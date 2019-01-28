from pydub import AudioSegment

from Plot import MyPlot
from config import *
import scipy
from scipy.fftpack import fft,rfft
import numpy as np
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from FilterBank import  filterBank

from Windows import HannWindow, HammingWindow


def convert_to_wav(filename):
    if os.path.exists(os.path.join(WORKING_DIRECTORY + '/files', filename)):
        filename, extension= filename.split('.')
        audio = AudioSegment.from_mp3(os.path.join(WORKING_DIRECTORY + '/files', filename + '.mp3'))
        audio.export(os.path.join(os.path.join(WORKING_DIRECTORY + '/files', filename + '.wav')), format='wav')
        os.remove(os.path.join(WORKING_DIRECTORY + '/files', filename + '.mp3'))


def preemphasize(data, alpha):
    """
    preemphasize signal with alpha i.e. apply filter xp(t) = x(t) - alpha * x(t-1) to it.
    :param data: signal to be preemphasized
    :param alpha: generally a value between 0.95 and 0.98
    :return: preemphasized signal
    """
    return np.append([data[0]], data[1:] - alpha * data[:-1], axis=0)



# speakerName = 'a'
file = 'a.mp3'
convert_to_wav(file)
sampleRate, data = read(os.path.join(WORKING_DIRECTORY + '/files', 'a.wav'))
# Convert stereo signal to mono
data = data.sum(axis=1)/2
# print(len(data) / sampleRate)
if SHOW_PLOTS:
    plt.figure(1)
    plt.subplot(211)
    p1 = MyPlot(len(data)/sampleRate, len(data), data)
    p1.x_label='t'
    p1.y_label='Signal value'
    p1.label = 'Not preemphasized'
    p1.plot()
if PREEMPHASIZE_ENABLED:
    data = preemphasize(data, ALPHA)
    if SHOW_PLOTS:
        plt.subplot(212)
        p2 = MyPlot(len(data)/sampleRate, len(data), data)
        p2.x_label='t'
        p2.y_label='Signal value'
        p2.label = 'Preemphasized'
        p2.plot()
        plt.show()

if SHOW_PLOTS:
    window = HannWindow(WINDOW_LENGTH)
    plt.figure(2)
    plt.subplot(211)
    p1 = MyPlot(WINDOW_LENGTH,1,window.window)
    p1.label = 'Hanning window'
    window = HammingWindow(WINDOW_LENGTH)
    plt.subplot(212)
    p2 = MyPlot(WINDOW_LENGTH, 1, window.window)
    p2.label = 'Hamming window'
    plt.show()

window = HannWindow(WINDOW_LENGTH)
#TODO: check windowed_data length
windowed_data = np.zeros((int(data.__len__()/(WINDOW_LENGTH-WINDOWING_DELAY)), WINDOW_LENGTH),dtype=float)
for i in range(0,data.__len__()-int(WINDOW_LENGTH/2), WINDOW_LENGTH-WINDOWING_DELAY):
    # print(data[i:i+WINDOW_LENGTH] * window)
    if len(data[i:i+WINDOW_LENGTH])<WINDOW_LENGTH:
        windowed_data[int(i/WINDOW_LENGTH)] = np.pad(data[i:i+WINDOW_LENGTH],(0, WINDOW_LENGTH - len(data[i:i+WINDOW_LENGTH])),mode='constant',constant_values=(0))
    else:
        windowed_data[int(i/WINDOW_LENGTH)] = data[i:i+WINDOW_LENGTH] * window.window
fourier = fft(windowed_data, n=FFTNUM)
modulus = np.zeros(fourier.__len__())
modulus = (np.absolute(fourier)**2) [:, 0:int(FFTNUM/2)] / FFTNUM
# print(modulus[0])
# plt.figure(1)
# p = MyPlot(256,1,modulus[45053])
# p.label = 'modulus'
# plt.show()
fb = filterBank(sampleRate)
fb = np.dot(modulus, fb.T)
fb = np.where(fb == 0.0, 10**-20, fb)
fb = 20 * np.log10(fb)
fb = scipy.fftpack.dct(fb,type=3,norm=None)


# file = open(,'w')
# file.write(fb)
np.savetxt(os.path.join(WORKING_DIRECTORY+'/feature_vectors',speakerName+'.dat'),fb)
print(fb[45053])
# print(fb[45053].shape)
# plt.figure(1)
# p = MyPlot(26,1,fb[45053])
# p.label = 'modulus'
# plt.show()
print(modulus.shape)

