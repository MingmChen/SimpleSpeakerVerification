from pydub import AudioSegment
from Plot import MyPlot
from config import *
import scipy
from scipy.fftpack import fft, rfft
import numpy as np
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from FilterBank import filterBank

from Windows import HannWindow, HammingWindow
from scipy import stats


def convert_to_wav(filename):
    if os.path.exists(filename):
        audio = AudioSegment.from_mp3(filename)
        filename = filename.split('.')[0]
        audio.export(filename + '.wav', format='wav')
        os.remove(filename + '.mp3')


def preemphasize(data, alpha):
    """
    preemphasize signal with alpha i.e. apply filter xp(t) = x(t) - alpha * x(t-1) to it.
    :param data: signal to be preemphasized
    :param alpha: generally a value between 0.95 and 0.98
    :return: preemphasized signal
    """
    return np.append([data[0]], data[1:] - alpha * data[:-1], axis=0)


# speakerName = 'a'

"""
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
"""



def extract_features(sampleRate, data, fileAddr, override_file=False):
    """
    Extract feature from data.
    :param sampleRate: sample rate of audio data
    :param data: audio data
    :param override_file: if True, it will analyze data and if feature files exist will override it.
                        if False, if the feature file exists it will skip this data and return immediately.
    :return:
    """
    # Convert stereo signal to mono
    if os.path.isfile(fileAddr) and not override_file:
        return
    if data.shape[-1] == 2:
        data = data.sum(axis=1) / 2
        # 1. Preemphasizing
    if PREEMPHASIZE_ENABLED:
        data = preemphasize(data,ALPHA)
        # 2. Windowing
    window_sample_num = int(WINDOW_LENGTH * sampleRate / 1000)
    window_overlap_num = int((WINDOW_LENGTH - WINDOWING_DELAY) * sampleRate / 1000)
    window = HannWindow(window_sample_num)
    windowed_data = np.zeros((int(data.__len__() / window_overlap_num), int(window_sample_num)), dtype=float)
    for i in range(0, data.__len__() - int(window_sample_num / 2), window_sample_num - window_overlap_num):
        if len(data[i:i + window_sample_num]) < window_sample_num:
            try:

                windowed_data[int(i / window_overlap_num)] = np.pad(data[i:i + window_sample_num],
                                                               (0, window_sample_num - len(data[i:i + window_sample_num])),
                                                               mode='constant', constant_values=(0)) * window.window
            except IndexError as e:
                print(fileAddr)
                print(e)
        else:
            try:
                windowed_data[int(i / window_overlap_num)] = data[i:i + window_sample_num] * window.window
            except ValueError as e:
                print(e)
    # 3.Calculate fourier transform of windows.
    fourier = fft(windowed_data, n=FFTNUM)
    #  4. Obtain power spectrum
    modulus = np.zeros(fourier.__len__())
    modulus = (np.absolute(fourier) ** 2)[:, 0:int(FFTNUM / 2)] / FFTNUM
    # print(modulus[0])
    # plt.figure(1)
    # p = MyPlot(256,1,modulus[45053])
    # p.label = 'modulus'
    # plt.show()
    # 5. Multiply in filterbank
    fb = filterBank(sampleRate)
    fb = np.dot(modulus, fb.T)
    fb = np.where(fb == 0.0, 10 ** -20, fb)
    fb = 20 * np.log10(fb)
    fb = scipy.fftpack.dct(fb, type=3, norm=None)
    np.savez(fileAddr,fb)
def read_file(fileName, userID):
    if  fileName.split('.')[-1]=='mp3':
        convert_to_wav(fileName)
        fileName += fileName.split('.')[0] + '.wav'
        sampleRate, data = read(fileName)
    elif fileName.split('.')[-1]=='wav':
        sampleRate, data = read(fileName)
    elif fileName.split('.')[-1]=='flac':
        import soundfile as sf
        data,sampleRate = sf.read(fileName)
    # sampleRate, data = read(fileName)
    if not os.path.isdir(os.path.join(WORKING_DIRECTORY + '/feature_vectors/' + str(userID))):
        os.makedirs(os.path.join(WORKING_DIRECTORY + '/feature_vectors/' + str(userID)))
    fileName = fileName.split('/')[-1].split('.')[0]+'.npz'
    extract_features(sampleRate,data,os.path.join(WORKING_DIRECTORY + '/feature_vectors/' + str(userID),fileName))

# read_file(os.path.join(WORKING_DIRECTORY+'/files','a'))
if FEATURE_EXTRACTION_ENABLE:
    users = os.listdir(WORKING_DIRECTORY+'/files')
    users = sorted(users,key=int)
    for user in users:
        directory =os.path.join(WORKING_DIRECTORY+'/files',user)

        # print(directory)
        # for directory in users:
        fileNames = []
        for path, subdirs, files in os.walk(directory):
            for name in files:
                if name.split('.')[-1] == 'flac':
                    fileNames.append(os.path.join(path,name))
        fileNames = fileNames[0:int(len(fileNames)*(1-TEST_SET_RATIO/100))]

        for file in fileNames:
            read_file(file, user)

if TRAIN_MODEL_ENABLE:
    # TODO: train model for each user.
    pass
