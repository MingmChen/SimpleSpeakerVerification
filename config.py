import os

WORKING_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ALPHA = 0.97
PREEMPHASIZE_ENABLED = True
WINDOW_LENGTH = 30  # in miliseconds. must be converted to samples when using.
WINDOWING_DELAY = 10  # in miliseconds. must be converted to samples when using.
SHOW_PLOTS = False
FFTNUM = 512  # Number of points for FFT. Usually a power of 2 and classically it's 512
FB_TYPE = 'triangular'  # Filter bank type.
FB_FILTER_NUM = 26    # Number of filters of filter-bank from this site:
# http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/

FB_UPPER_FREQUENCY = 8000  # Upper frequency for filter bank (in Hz). P.S.: it should be half of sampling frequency of input.
FB_LOWER_FREQUENCY = 300
LOG_ENABLED = False
TEST_SET_RATIO = 15  # % of data will be used as test set
FEATURE_EXTRACTION_ENABLE = False
TRAIN_MODEL_ENABLE = True
