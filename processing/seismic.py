# matplotlib.use('Agg')
# plt.rcParams['figure.dpi'] = 400
# plt.rcParams['figure.figsize'] = [10,10]

import numpy as np


import traceback
from processing.signal import *
from processing.io import *
from processing.PlotTools import *
from processing.misc import *


# Removed because caused problems when using the API
# plt.style.use(["science", "notebook", "grid"])

# TODO:
# Add anti-triggering function
# Add rotate function
# Select windows with at-least method
# Select windows options "from" and "to"
# Add filtering function
# Add rotated H/V
# Horizontal particle motion


def hv_ratio(amp_north, amp_vertical, amp_east, freq, fmin=0.1, fmax=10):
    """Calculates the H/V spectral ratio

    Args
    ----
        - amp_north (ndarray): North-component amplitude spectrum
        - amp_vertical (ndarray): Vertical-component amplitude spectrum
        - amp_east (ndarray): East-component amplitude spectrum

    Returns
    -------
        - tuple: tuple containing the H/V mean ratio and the H/V for each window
    
    Changelog
    ---------
        - 09/SEP/23:\n
            --> Added freq in the return variables for use in the plot function, as it's already cropped to the desired shape
        - 11/SEP/2023:\n
            --> Added a loop to check if the cutoff frequencies are inside the specified interval

    """

        # Encuentra el índice del elemento más cercano a fmin y fmax
    fmin_loc = np.abs(freq-fmin)
    fmin_loc = np.argmin(fmin_loc)
    fmax_loc = np.abs(freq-fmax)
    fmax_loc = np.argmin(fmax_loc)

    # Comprueba que el valor de freq no sea menor o mayor que fmin y fmax
    while freq[fmin_loc] < fmin:
        fmin_loc += 1

    while freq[fmax_loc] > fmax:
        if freq[fmax_loc] == float(fmax):
            break
        fmax_loc -= 1

    
    # # Corta el vector de frecuencias desde fmin hasta fmax
    freq = np.array(freq[fmin_loc:fmax_loc])

    amp_north = np.array(
        np.split(ary=amp_north, 
                 indices_or_sections=[fmin_loc, fmax_loc], 
                 axis=-1)[1]
                 )
    amp_vertical = np.array(
        np.split(ary=amp_vertical, 
                 indices_or_sections=[fmin_loc, fmax_loc], 
                 axis=-1)[1])
    amp_east = np.array(
        np.split(ary=amp_east, 
                 indices_or_sections=[fmin_loc, fmax_loc], 
                 axis=-1)[1])

    amp_horizontal = []
    for n, e in zip(amp_north, amp_east):
        if isinstance(n, float) == True and isinstance(e, float) == True:
            amp_horizontal.append(
                np.sqrt((n**2 + e**2)/2))
        else:
            amp_horizontal.append(
                np.sqrt((n.astype(float)**2 + e.astype(float)**2)/2))

    amp_horizontal = np.array(amp_horizontal, dtype=object)

    mask = amp_vertical != 0

    try:
        HV = np.divide(amp_horizontal, amp_vertical, out=np.full_like(
            amp_horizontal, np.nan), where=mask)
    except ZeroDivisionError:
        print('\nError al calcular H/V. Existen valores NaN en los vectores')
        quit()

    if HV.ndim != 1:
        HV_mean = np.mean(HV, axis=0)
    else:
        HV_mean = HV

    return HV_mean, HV, freq


def process_hvsr(data, dt, win_len, taper_win, smooth_coeff, fftmin, fftmax, hvmin, hvmax):
    """Calculates H/V Spectral Ratio from acceleration data, either from array or file.


    Args:
    -----
        - data (NDArray or string): Path of the file that contains the data (str) or variable with data (NDArray)

        - dt (float): Sampling period of the signal

        - win_len (float): Length in seconds of the analysis window

        - taper_win (str): Window type for taper. Valid windows are those in the scipy.signal.windows module, but only the ones that does not take any special parameters: {'barthann','bartlett','blackman','blackmanharris','bohman','boxcar','chebwin','cosine','exponential','flattop','hamming','hann','lanczos','nuttall','parzen','taylor','triang','tukey'}

        - smooth_coeff (float): Smooth coefficient of the Konno-Ohmachi smoothing function.
        - fftmin (float): Minimum frequency for the FFT spectrum
        - fftmax (float): Maximum frequency for the FFT spectrum
        - hvmin (float): Minimum frequency for the H/V spectrum
        - hvmax (float): Maximum frequency for the FFT spectrum

    Returns:
    --------
        ndarray: Array with shape (3,n) in the form (HV_mean, HV, freq)
    """    
    """
    Workflow:
    1. Read file (specify)
    2. Separate into number of windows
    3. Crop signal so each window has the same length
    4. Taper signal
    5. Calculate FFT
    6. Smooth spectrum
    7. Calculate H/V
    """
    fs = 1/dt

    # data_windowed returns a 3-element tuple (N, V, E) and window number
    # return [north, vertical, east], window_number
    data_windowed = window(win_len, data, fs)
    # print('Data windowed')

    try:
        north = data_windowed[0]
        vertical = data_windowed[1]
        east = data_windowed[2]
    except IndexError as ie:
        traceback.print_exc()
        print('Windowed array has more than 3 sub-arrays')
    
    # ------ #
    # TODO: This process is already performed in the window function
    # taper returns a 3-element tuple (N-tapered, V-tapered, E-tapered)
    # data_tapered = taper(north, vertical, east, taper_win)

    # try:
    #     north = data_tapered[0]
    #     vertical = data_tapered[1]
    #     east = data_tapered[2]
    # except IndexError as ie:
    #     traceback.print_exc()
    #     print('Tapered array has more than 3 sub-arrays')
    # print('\nData tapered')
    # ------ #


    # spectrum returns a 4-element tuple (NFFT, VFFT, EFFT, Frequency)
    # return fftnorth, fftvertical, ffteast, freq
    data_fft, freq = spectrum(north, vertical, east, dt, fmin=fftmin, fmax=fftmax)

    try:
        north = data_fft[0]
        vertical = data_fft[1]
        east = data_fft[2]
    except IndexError as ie:
        traceback.print_exc()
        print('FFT array has more than 4 sub-arrays')
    # print('\nFFT done')


    # konnoohmachi_smoothing_opt returns one array with the smoothed data
    print('Smoothing... this may take a while')
    data_smoothed = konnoohmachi_smoothing_opt(data_fft, freq, smooth_coeff)

    try:
        north = data_smoothed[0]
        vertical = data_smoothed[1]
        east = data_smoothed[2]
    except IndexError as ie:
        traceback.print_exc()
        print('Smoothed array has more than 3 sub-arrays')
    print('Smoothing done')


    # HV returns a 3-element tuple (HV mean, HV, Frequency)
    # return HV_mean, HV, freq
    data_hv = hv_ratio(north, vertical, east, freq, fmin=hvmin, fmax=hvmax)

    try:
        HV_mean = data_hv[0]
        HV = data_hv[1]
        freq = data_hv[2]
    except IndexError as ie:
        traceback.print_exc()
        print('HV array has more than 3 sub-arrays')
    print('Spectral ratio calculated!')


    return HV_mean, HV, freq

def hvsr(acc_data, dt: float, win_len: float, file_type=None, taper_window='cosine', smooth_bandwidth=40.0, fftmin=0.1, fftmax=50.0, hvmin=0.1, hvmax=10.0):
    """Calculates H/V Spectral Ratio given the acceleration data with north, east and vertical components

    Args:
    -----
        - acc_data (ndarray, str): Path of the file that contains the data (str) or variable with data (NDArray)

        - file_type (str, optional): File type. Valid options are {'cires','ascii','mseed','miniseed','sac'}. Defaults to None.

        - dt (float): Sampling period of the signal

        - win_len (float): Length in seconds of the analysis window

        - taper_window (str, optional): Window type for taper. Valid windows are those in the scipy.signal.windows module, but only the ones that does not take any special parameters: {'barthann','bartlett','blackman','blackmanharris','bohman','boxcar','chebwin','cosine','exponential','flattop','hamming','hann','lanczos','nuttall','parzen','taylor','triang','tukey'} Defaults to 'cosine'.

        - smooth_bandwidth (float, optional): Smooth coefficient of the Konno-Ohmachi smoothing function. Defaults to 40.

        - fftmin (float, optional): Minimum frequency for the FFT spectrum. Defaults to 0.1.
        - fftmax (float, optional): Maximum frequency for the FFT spectrum. Defaults to 50.
        - hvmin (float, optional): Minimum frequency for the H/V spectrum. Defaults to 0.1.
        - hvmax (float, optional): Maximum frequency for the H/V spectrum. Defaults to 10.

    Returns:
    --------
        ndarray: Array with shape (3,n) in the form (HV_mean, HV, freq)
    """    

    
    # if isinstance(acc_data, str):
    if file_type is not None:
        # checks if acc_data is path-like
        try:
            if file_type.lower() == 'cires':
                data = read_cires(acc_data)
            elif file_type.lower() == 'ascii':
                data = read_file(acc_data, skiprows=0)
            elif file_type.lower() == 'mseed' or file_type.lower() == 'miniseed':
                data = read_mseed(acc_data)
            elif file_type.lower() == 'sac':
                data = read_sac(acc_data)
        except FileNotFoundError as fnf:
            traceback.print_exc()

        hvmean, hv, freq = process_hvsr(data=data, dt=dt, win_len=win_len, taper_win=taper_window, smooth_coeff=smooth_bandwidth, fftmin=fftmin, fftmax=fftmax, hvmin=hvmin, hvmax=hvmax)
    else:
        # if acc_data is a variable (override reading the file)
        hvmean, hv, freq = process_hvsr(data=acc_data, dt=dt, win_len=win_len, taper_win=taper_window, smooth_coeff=smooth_bandwidth, fftmin=fftmin, fftmax=fftmax, hvmin=hvmin, hvmax=hvmax)
    
    return hvmean, hv, freq
