# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt.rcParams['figure.dpi'] = 400
# plt.rcParams['figure.figsize'] = [10,10]
from matplotlib.patches import Rectangle

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import detrend
from scipy.signal import windows

from obspy.signal.util import smooth
from obspy import read
from time import perf_counter
import pykooh  # https://github.com/arkottke/pykooh


def taper(north, vertical, east, type):

    """Taper the signal as to avoid continuity problems with FFT

    :warning: **Not longer in use**, will be removed

    Args:
    ----
        - north (ndarray): North component of the acceleration data
        - vertical (ndarray): Vertical component of the acceleration data
        - east (ndarray): East component of the acceleration data
        - type (str): Window type to apply the taper. Available windows are {'barthann','bartlett','blackman','blackmanharris','bohman','boxcar','chebwin','cosine','exponential','flattop','hamming','hann','lanczos','nuttall','parzen','taylor','triang','tukey'}

    Returns:
        tuple: North, vertical and east components tapered. Also includes the window vector

    Changelog:
        - 10/OCT/23:\n
            --> Added removal warning

    
    """
    north = np.array(north)
    vertical = np.array(vertical)
    east = np.array(east)

    try:
        w = windows.get_window(type, north.shape[-1])
    except ValueError:
        print('Ventana no válida')
        type = input('Ventana: ')

    north = north*w
    vertical = vertical*w
    east = east*w

    return north, vertical, east, w

def window(window_length: float, data, fs: float, taper=True):
    """Split the signal into the given number of sections

    Args:
    ----
        sections (float): Length of analysis window in seconds
        data (ndarray): Signal data to be split into sections
        fs (float): Sampling frequency
        taper (bool): Whether to apply tapering or not. Defaults to True.

    Returns:
    -------
        tuple: Tuple of north, vertical, east and window number
    
    Changelog:
    ---------
        - 13/SEP/2023:\n
            --> Added cropping function to keep equal dimensions
        - 24/OCT/2023:\n
            --> Added tapering function set to True by default
        - 25/OCT/2023:\n
            --> Changed the function so it's compatible with any array size
    """
    # N = len(data[0])

    N = data.shape[-1]
    section_length = window_length * fs  # número de muestras por sección
    window_number = np.floor(N/section_length)  # numero de ventanas
    # print(f'Número de ventanas: {int(window_number)}')

    if data.ndim != 1: # Si se está tratando con múltiples vectores al mismp tiempo (e.g. múltiples canales de un sismograma)
        data_split = []
        for arr in data:

            # Corta los vectores en un numero especificado de ventanas
            # La función array_split genera vectores de diferentes longitudes, por lo que hay que cortarlos todos a la menor longitud
            arr_split = np.apply_along_axis(func1d = lambda t: np.array_split(t, window_number),
                                            axis=-1,
                                            arr=arr)
            
            # Calcula la longitud menor de los arreglos
            mindim = min([len(i) for i in arr_split])

            # Corta todos los vectores a la menor longitud
            arr_split = [subarr[:mindim] for subarr in arr_split]

            # Añade los vectores cortados a una lista
            data_split.append(arr_split)

        # Convierte la lista en un numpy array
        data_split = np.array(data_split)
    else:
        data_split = np.array_split(data, window_number)
        mindim = min(len(arr) for arr in data_split)
        data_split = np.array([arr[:mindim] for arr in data_split])


    if not taper:
        return data_split
    else:
        window = windows.cosine(data_split.shape[-1])
        data_split_taper = np.apply_along_axis(lambda t: window*t,
                                         axis=-1,
                                         arr=data_split)
        return data_split_taper

def spectrum(north, vertical, east, dt, fmin=0.1, fmax=50):
    """Calculates the right-hand-side amplitude spectrum of the signal

    Args
    ----
        - north (ndarray): North-component signal
        - vertical (ndarray): Vertical-component signal
        - east (ndarray): East-component signal
        - dt (float): time precision
        - fmin (float): minimum frequency to calculate the FFT from. Defaults to 0.1
        - fmax (float): maximum frequency to calculate the FFT from. Defaults to 50

    Returns
    -------
        - tuple: Tuple that contains the north, vertical and east amplitude spectrums
        - array: Frequency array

    Changelog
    ---------
        - 09/SEP/2023: \n
            --> Changed lists comprehensions to np.apply_along_axis() function for simplicity and performance.\n
            --> Added code to crop the FFT and frequency arrays to the specified fmin and fmax
        - 11/SEP/2023: \n
            --> Added a loop to check whether the cutoff frequencies are inside the right interval
        - 25/SEP/2023:\n
            --> Changed the output type to numpy arrays. Now there are two outputs, one with the FFT data and another with the frequency array
        - 25/OCT/2023:\n
            --> Changed function name from `rhs_spectrum` to `spectrum`
    """
    # TODO: make it work with arrays of any dimension

    north = np.array(north)
    vertical = np.array(vertical)
    east = np.array(east)

    if north.ndim == 1:
        N = len(north)
    else:
        N = len(north[0])  # longitud de cada seccion
    freq = rfftfreq(N, dt)

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

    # Cálculo y normalización de los espectros de Fourier
    if north.ndim != 1 and vertical.ndim != 1 and east.ndim != 1:
        fftnorth = np.apply_along_axis(
            func1d=lambda t: np.abs(rfft(t)/(N/2)), 
            axis=-1, 
            arr=north)
        fftvertical = np.apply_along_axis(
            func1d=lambda t: np.abs(rfft(t)/(N/2)), 
            axis=-1, 
            arr=vertical)
        ffteast = np.apply_along_axis(
            func1d=lambda t: np.abs(rfft(t)/(N/2)), 
            axis=-1, 
            arr=east)
    else:
        fftnorth = np.abs(rfft(north)/(N/2))
        fftvertical = np.abs(rfft(vertical)/(N/2))
        ffteast = np.abs(rfft(east)/(N/2))

    # Corta el vector de frecuencias desde fmin hasta fmax
    freq = freq[fmin_loc:fmax_loc]

    fftnorth = np.array(
        np.split(
            ary=fftnorth, 
            indices_or_sections=[fmin_loc, fmax_loc], 
            axis=-1)[1])
    fftvertical = np.array(
        np.split(
            ary=fftvertical, 
            indices_or_sections=[fmin_loc, fmax_loc], 
            axis=-1)[1])
    ffteast = np.array(
        np.split(
            ary=ffteast, 
            indices_or_sections=[fmin_loc, fmax_loc], 
            axis=-1)[1])


    return np.array([fftnorth, fftvertical, ffteast]), np.array(freq)

def standard_smoothing(fftnorth, fftvertical, ffteast, smoothie):
    """Smooths the data by calculating a moving average

    Args
    ----
        - fftnorth (ndarray): North-component amplitude spectrum
        - fftvertical (ndarray): Vertical-component amplitude spectrum
        - ffteast (ndarray): East-component amplitude spectrum
        - smoothie (int): Degree of smoothing

    Returns
    -------
        tuple: tuple containing the smoothed data
    """
    amp_north = []
    amp_east = []
    amp_vertical = []
    s = perf_counter()
    for i, j, k in zip(fftnorth, fftvertical, ffteast):
        amp_north.append(smooth(i, smoothie=smoothie))
        amp_vertical.append(smooth(j, smoothie=smoothie))
        amp_east.append(smooth(k, smoothie=smoothie))
    amp_north = np.array(amp_north, dtype=object)
    amp_east = np.array(amp_east, dtype=object)
    amp_vertical = np.array(amp_vertical, dtype=object)

    e = perf_counter()
    # print(f'{round(e-s, 4)} s')

    return amp_north, amp_vertical, amp_east

def konnoohmachi_smoothing(fftnorth, fftvertical, ffteast, freq, bandwidth=40):
    """Smooths the data using Konno-Ohmachi (1998) algorithm

    Args
    ----
        - fftnorth (ndarray): North-component amplitude spectrum
        - fftvertical (ndarray): Vertical-component amplitude spectrum
        - ffteast (ndarray): East-component amplitude spectrum
        - freq (ndarray): Frequency vector
        - bandwidth (int): Strength of the filter. A lower bandwidth is a stronger smoothing.

    Returns
    -------
        - tuple: smoothed spectrum

    Notes
    -----
        # TODO:
        Maybe will be removed in future versions
    """
    north_smooth = []
    east_smooth = []
    vertical_smooth = []
    count = 0
    s = perf_counter()  # Cuenta el tiempo de ejecución del ciclo
    for i, j, k in zip(fftnorth, fftvertical, ffteast):
        # WITH PYKOOH (better performance than Obspy)
        north_smooth.append(pykooh.smooth(freq, freq, i, b=bandwidth))
        vertical_smooth.append(pykooh.smooth(freq, freq, j, b=bandwidth))
        east_smooth.append(pykooh.smooth(freq, freq, k, b=bandwidth))

        # print(f'Ventana {count} lista')

    # Convierte lista en numpy array
    north_smooth = np.array(north_smooth, dtype=object)
    vertical_smooth = np.array(vertical_smooth, dtype=object)
    east_smooth = np.array(east_smooth, dtype=object)

    e = perf_counter()
    print(f'{round(e-s, 4)} s transcurridos')

    return north_smooth, vertical_smooth, east_smooth


def konnoohmachi_smoothing_opt(data, freq, bandwidth=40, axis=-1):
    """Smooths the data using Konno-Ohmachi (1998) algorithm.\n
    Optimized version with numpy vectorization and cython for smoothing.\n
    Up to 2x faster than normal version when already allocated in memory

    Args
    ----
        - data (ndarray): Data vector to be smoothed
        - freq (ndarray): Frequency vector
        - bandwidth (int): Strength of the filter. A lower bandwidth is a stronger smoothing. Defaults to 40.
        - axis (int): Axis along which the data will be smoothed. Defaults to -1

    Returns
    -------
        - ndarray: smoothed data
    
    Changelog
    ---------
    - 10/SEP/23:\n
        --> Added the function
    - 24/SEP/23:\n
        --> Added support for arrays of any dimension
    """

    data_smooth = np.apply_along_axis(
                    func1d=lambda t: pykooh.smooth(freq, freq, t, b=bandwidth, use_cython=True), 
                    arr=data, 
                    axis=axis
                )
    return data_smooth

def konnoohmachi_matlab(signal, freq_array, smooth_coeff):
    """
    Function taken from \n
    Hamdullah Livaoglu (2023). Konno-Ohmachi smoothing function for ground motion spectra (https://www.mathworks.com/matlabcentral/fileexchange/68205-konno-ohmachi-smoothing-function-for-ground-motion-spectra), MATLAB Central File Exchange. Retrieved septiembre 12, 2023.

    Notes
    -----
    Added 12/SEP/2023
    
    """
    x = signal
    f = freq_array
    f_shifted = f / (1 + 1e-4)
    L = len(x)
    y = np.zeros(L)

    for i in range(L):
        if i != 0 and i != L - 1:
            z = f_shifted / f[i]
            w = ((np.sin(smooth_coeff * np.log10(z)) / smooth_coeff) / np.log10(z)) ** 4
            w[np.isnan(w)] = 0
            y[i] = np.sum(w * x) / np.sum(w)
    
    y[0] = y[1]
    y[-1] = y[-2]
    
    return y
