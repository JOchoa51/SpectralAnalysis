# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import detrend
from scipy.signal import windows
from obspy.signal.util import smooth
from obspy import read
from time import perf_counter
import pykooh  # https://github.com/arkottke/pykooh
import os
import datetime
import colorsys
import traceback


# Removed because caused problems when using the API
# plt.style.use(["science", "notebook", "grid"])


def read_sac(name, name2, name3: str):
    """Read MSEED files using the ´read´ function of OpsPy

    Args
    ----
        name (str): Route of the file

    Returns
    -------
        tuple: A tuple of north, vertical and east components
    """
    st1 = read(name)
    st2 = read(name2)
    st3 = read(name3)

    D1 = detrend(np.array(st1[0].data) - np.mean(np.array(st1[0].data)))
    D2 = detrend(np.array(st2[0].data) - np.mean(np.array(st2[0].data)))
    D3 = detrend(np.array(st2[0].data) - np.mean(np.array(st3[0].data)))

    dimmin = np.min([len(D1), len(D2), len(D3)])

    # Corte de las señales a la menor dimensión encontrada
    D1 = D1[:dimmin]
    D2 = D2[:dimmin]
    D3 = D3[:dimmin]

    return D1, D2, D3


def read_mseed(name: str):
    """Read MSEED files using the ´read´ function of OpsPy

    Args
    ----
        name (str): Route of the file

    Returns
    -------
        tuple: A tuple of north, vertical and east components
    """
    try:
        st = read(name)
    except FileNotFoundError:
        print('Archivo no encontrado. Intenta de nuevo: ')
        st = read(name)

    print(st[0].stats)
    n = int(input('Señal norte: '))
    e = int(input('Señal este: '))
    v = int(input('Señal vertical: '))
    N = detrend(np.array(st[n-1].data))
    V = detrend(np.array(st[v-1].data))
    E = detrend(np.array(st[e-1].data))

    dimmin = np.min([len(N), len(V), len(E)])

    # Corte de las señales a la menor dimensión encontrada
    N = N[:dimmin]
    V = V[:dimmin]
    E = E[:dimmin]

    return N, V, E


def read_file(name: str):
    """Read data from ASCII file, i.e. TXT files

    Args
    ----
        name (str): Route of the file

    Returns
    -------
        tuple: A tuple of north, vertical and east components
    """
    N, V, E = np.loadtxt(name).transpose()
    N = detrend(np.array(N)) - np.mean(N)
    V = detrend(np.array(V)) - np.mean(V)
    E = detrend(np.array(E)) - np.mean(E)

    return N, V, E


def read_cires(name: str):
    """Function specifically designed to read data from accelerograms of CIRES

    Args
    ----
        name (str): Route of the file

    Returns
    -------
        tuple: A tuple of north, vertical and east components
    """
    try:
        with open(name, "r") as f:
            north = []
            vertical = []
            east = []
            header = []
            count = 0
            for line in f:
                count += 1
                if line.startswith('NOMBRE DE LA ESTACION'):
                    station_name = 'Nombre de la estación: ' + line.split(":")[1]
                if line.startswith('CLAVE DE LA ESTACION'):
                    station_key = 'Clave de la estación: ' +  line.split(":")[1]
                    station_NamePlusKey = station_name + station_key
                    # print(station_NamePlusKey)
                # if 
                if line.startswith('HORA DE LA PRIMERA MUESTRA'):
                    initial_time = line.split(":")[1]
                    # print(initial_time)
                if count < 110:
                    header.append(line)
                    # Imprime encabezado
                    # print(line.split("\n")[0])
                    # if line.split("\n")[0].startswith('NOMBRE DE LA ESTACION'):
                    #     print(line)
                else:
                    try:
                        north.append(float(line.split()[0]))
                        vertical.append(float(line.split()[1]))
                        east.append(float(line.split()[2]))
                    except ValueError:
                        s = line.split()[1].split("-")
                        if len(s) == 3:
                            vertical.append(-float(s[1]))
                            east.append(-float(s[2]))
                        elif len(s) == 2:
                            vertical.append(float(s[0]))
                            east.append(-float(s[1]))
    except Exception as e:
        print(e)
    north = detrend(np.array(north))
    vertical = detrend(np.array(vertical))
    east = detrend(np.array(east))

    return north, vertical, east, header


def time_sequence(seconds, fs):
    start_time = datetime.datetime.min.time()  # Start time as midnight (00:00:00)
    end_time = datetime.datetime.combine(
        datetime.date.min, start_time) + datetime.timedelta(seconds=seconds)  # End time
    time_increment = datetime.timedelta(milliseconds=fs/10)  # Time increment of 0.01 seconds

    current_time = datetime.datetime.combine(datetime.date.min, start_time)

    time_sequence = []

    while current_time <= end_time:
        time_sequence.append(current_time.time())
        current_time += time_increment

    for time in time_sequence:
        # Format time as hh:mm:ss.ms
        time_str = time.strftime("%H:%M:%S.%f")[:-3]

    return time_sequence[:-1]


def plot_signal(north, vertical, east, dt, name):
    """Plots the three components of the provided acceleration data

    Args:
        north (ndarray): North component of the acceleration data
        vertical (ndarray): Vertical component of the acceleration data
        east (ndarray): East component of the acceleration data
        dt (float): Sampling period of the signal
        name (str): Name of the seismic station or path to the original file. Title of the plot

    Returns:
        figure: matplotlib figure
    """    
    min_dim = min([len(north), len(vertical), len(east)])

    # crop the north, vertical and east array to min_dim length
    north = north[-min_dim:]
    vertical = vertical[-min_dim:]
    east = east[-min_dim:]

    n = len(north)
    time = np.arange(0, n*dt, dt)
    time = time[-min_dim:]

    maxvalue = np.max(np.maximum(np.abs(north), np.abs(east)))*1.2
    # minvalue = np.min(np.minimum(north, east))*1.2
    minvalue = -maxvalue

    nombre = os.path.basename(name)[:4]
    time_min = time/60
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 10))
    ax = axes[0]
    ax.plot(time, north, lw=0.5, color='k')
    ax.fill_between(time, north, where=(north > 0), color='black')
    # ax.set_title('North', fontsize=12)
    ax.set_ylim(minvalue, maxvalue)
    ax.set_ylabel(f'{nombre} N', rotation=0, labelpad=30)
    ax.set_yticks([])

    ax = axes[1]
    ax.plot(time, vertical, lw=0.5, color='k')
    ax.fill_between(time, vertical, where=(vertical > 0), color='black')
    # ax.set_title('Vertical', fontsize=12)
    ax.set_ylim(minvalue, maxvalue)
    ax.set_ylabel(f'{nombre} Z', rotation=0, labelpad=30)
    ax.set_yticks([])

    ax = axes[2]
    ax.plot(time, east, lw=0.5, color='k')
    ax.fill_between(time, east, where=(east > 0), color='black')
    # ax.set_title('East', fontsize=12)
    ax.set_ylim(minvalue, maxvalue)
    ax.set_ylabel(f'{nombre} E', rotation=0, labelpad=30)
    ax.set_yticks([])

    fig.supxlabel('Time [s]', fontsize=12)
    plt.tick_params('x', labelsize=12)
     
    plt.subplots_adjust(hspace=0.5)
    
    # # plt.show()

    return fig


def crop_signal(north, vertical, east):

    # Cálculo de la dimensión menor de las señales
    # Utilidad opcional.
    dimmin = np.min([[len(no) for no in north] +
                    [len(e) for e in east] +
                    [len(v) for v in vertical]])

    # Corte de las señales a la menor dimensión encontrada
    for ni, ei, vi in zip(range(len(north)), range(len(east)), range(len(vertical))):
        north[ni] = north[ni][:dimmin]
        east[ei] = east[ei][:dimmin]
        vertical[vi] = vertical[vi][:dimmin]

    return north, vertical, east


def taper(north, vertical, east, type):

    """Taper the signal as to avoid continuity problems with FFT

    Args:
    ----
        - north (ndarray): North component of the acceleration data
        - vertical (ndarray): Vertical component of the acceleration data
        - east (ndarray): East component of the acceleration data
        - type (str): Window type to apply the taper. Available windows are {'barthann','bartlett','blackman','blackmanharris','bohman','boxcar','chebwin','cosine','exponential','flattop','hamming','hann','lanczos','nuttall','parzen','taylor','triang','tukey'}

    Returns:
        tuple: North, vertical and east components tapered. Also includes the window vector
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


def window(window_length: float, data, fs: float):
    """Split the signal into the given number of sections

    Args:
    ----
        sections (float): Length of analysis window in seconds
        data (ndarray): Signal data to be split into sections

    Returns:
    -------
        tuple: Tuple of north, vertical and east components, each one with the specified number of sections
    """
    N = len(data[0])
    section_length = window_length * fs  # número de muestras por sección
    window_number = np.floor(N/section_length)  # numero de ventanas
    # print(f'Número de ventanas: {int(window_number)}')

    north = np.array_split(data[0], window_number)
    vertical = np.array_split(data[1], window_number)
    east = np.array_split(data[2], window_number)

    return [north, vertical, east], window_number


def plot_windows(data, window_length, dt, name):
    """Plot the signal with the windows overlayed as to see them

    Args:
    -----
        - data (ndarray): Array with the acceleration data. It must be a 2D array with three sub-arrays. 
        #TODO: give support for multiple array shapes
        - window_length (float): Length of the analysis window in seconds
        - dt (float): Sampling period of the signal
        - name (str): Name of the seismic station or path to the original analyzed file.

    Returns:
    --------
        figure: matlplotlib figure
    """
    north = data[0]
    vertical = data[1]
    east = data[2]
    fs = 1/dt
    N = len(north)
    section_length = window_length * fs  # número de muestras por sección
    try:
        window_number = int(np.floor(N/section_length))  # numero de ventanas
    except ZeroDivisionError:
        window_number = 1  # numero de ventanas

    # list of window_number html colors from red to purple
    start_color = '#fc0400'  # Red
    end_color = '#0313fc'  # Purple
    colors = [start_color]  # Start with the red color

    # Calculate the hue values for the rainbow spectrum
    start_hue = colorsys.rgb_to_hsv(int(start_color[1:3], 16), int(start_color[3:5], 16), int(start_color[5:7], 16))[0]
    end_hue = colorsys.rgb_to_hsv(int(end_color[1:3], 16), int(end_color[3:5], 16), int(end_color[5:7], 16))[0]

    for i in range(1, window_number - 1):
        # Interpolate the hue values between start and end hues
        hue = start_hue + (i / (window_number - 1)) * (end_hue - start_hue)

        # Convert the hue value back to RGB color
        rgb_color = colorsys.hsv_to_rgb(hue, 1, 1)

        # Convert the RGB color to HTML hexadecimal format
        color = '#{0:02X}{1:02X}{2:02X}'.format(int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))
        colors.append(color)

    colors.append(end_color)  # Append the purple color

    
    nombre = os.path.basename(name)[:4]
    time = np.linspace(0, len(north)*dt, len(north))

    box_len = int(section_length)
    x_min = np.array([i*box_len for i in range(window_number)])*dt
    x_max = np.array([xmin + np.diff(x_min)[0] for xmin in x_min])

    y_max = np.max(np.maximum(np.abs(north), np.abs(east)))*1.2
    # y_min = np.min(np.minimum(north, east))*1.2
    y_min = -y_max


    fig, axes = plt.subplots(3, 1, sharex=True)
    ax = axes[0]
    ax.plot(time, north, lw=0.5, color='k')
    ax.fill_between(time, north, where=(north > 0), color='black')
    for xmx, xmn, c in zip(x_max, x_min, colors):
        ax.add_patch(Rectangle((xmn, y_min), xmx-xmn, y_max-y_min, alpha=0.4, facecolor=c, edgecolor='k'))
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel(f'{nombre} N', rotation=0, labelpad=30)
    ax.set_yticks([])
    

    ax = axes[1]
    ax.plot(time, vertical, lw=0.5, color='k')
    ax.fill_between(time, vertical, where=(vertical > 0), color='black')
    for xmx, xmn, c in zip(x_max, x_min, colors):
        ax.add_patch(Rectangle((xmn, y_min), xmx-xmn, y_max-y_min, alpha=0.4, facecolor=c, edgecolor='k'))
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel(f'{nombre} Z', rotation=0, labelpad=30)
    ax.set_yticks([])

    ax = axes[2]
    ax.plot(time, east, lw=0.5, color='k')
    ax.fill_between(time, east, where=(east > 0), color='black')
    for xmx, xmn, c in zip(x_max, x_min, colors):
        ax.add_patch(Rectangle((xmn, y_min), xmx-xmn, y_max-y_min, alpha=0.4, facecolor=c, edgecolor='k'))
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel(f'{nombre} E', rotation=0, labelpad=30)
    ax.set_yticks([])
    
    fig.supxlabel('Time [s]', fontsize=12)
    plt.tick_params('x', labelsize=12)

    plt.subplots_adjust(hspace=0.5)
    # # plt.savefig(f'{name[:-4]}-SISMOGRAMA.png', dpi=400)
    
    # # plt.show()

    return fig


def plot_signal_windowed(north, vertical, east, dt, name, window_type='cosine'):
    """Plot each individual analysis window

    Args:
    ----
        - north (ndarray): North component of the acceleration data
        - vertical (ndarray): Vertical component of the acceleration data
        - east (ndarray): East component of the acceleration data
        - dt (float): Sampling period of the signal
        - name (str): Name of the seismic station or path of the analyzed file
        - window_type (str, optional): Window type for tapering. Available windows are: {'barthann','bartlett','blackman','blackmanharris','bohman','boxcar','chebwin','cosine','exponential','flattop','hamming','hann','lanczos','nuttall','parzen','taylor','triang','tukey'}. Defaults to 'cosine'.

    Returns:
    -------
        figure: matplotlib figure
    """    
    n = len(north[0])
    maxvalue = np.max(np.maximum(np.abs(north), np.abs(east)))*1.2
    # minvalue = np.min(np.minimum(north, east))*1.2
    minvalue = -maxvalue
    time = np.arange(0, n*dt, dt)
    nombre = os.path.basename(name)[:4]

    fig, axes = plt.subplots(3, 1, sharex=True)
    ax = axes[0]
    ax.plot(window_type, color='k', lw=1)
    count = 0
    for i in north:
        count += 1
        ax.plot(time, i, label=str(count), lw=0.75)
    count = 0
    # ax.set_ylim(y_min, y_max)
    ax.set_ylim(minvalue, maxvalue)
    ax.set_ylabel(f'{nombre} Z', rotation=0, labelpad=30)
    ax.set_yticks([])
    
    ax = axes[1]
    ax.plot(window_type, color='k', lw=1)
    for i in east:
        count += 1
        ax.plot(time, i, label=str(count), lw=0.75)
    count = 0
    ax.set_ylim(minvalue, maxvalue)
    ax.set_ylabel(f'{nombre} Z', rotation=0, labelpad=30)
    ax.set_yticks([])

    ax = axes[2]
    ax.plot(window_type, color='k', lw=1)
    for i in vertical:
        count += 1
        ax.plot(time, i, lw=0.75)
    count = 0
    ax.set_ylim(minvalue, maxvalue)
    ax.set_ylabel(f'{nombre} Z', rotation=0, labelpad=30)
    ax.set_yticks([])
    
    fig.supxlabel('Time [s]', fontsize=12)
    plt.tick_params('x', labelsize=12)
    # plt.legend()
    plt.subplots_adjust(hspace=0.5)
    # plt.show()

    return fig


def rhs_spectrum(north, vertical, east, dt, fmin=0.1, fmax=50):
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

    Changelog
    ---------
        - 09/SEP/2023: \n
            --> Changed lists comprehensions to np.apply_along_axis() function for simplicity and performance.\n
            --> Added code to crop the FFT and frequency arrays to the specified fmin and fmax
        - 11/SEP/2023: \n
            --> Added a loop to check whether the cutoff frequencies are inside the right interval
    """
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
    fftnorth = np.apply_along_axis(lambda t: np.abs(rfft(t)/(N/2)), axis=-1, arr=north)
    fftvertical = np.apply_along_axis(lambda t: np.abs(rfft(t)/(N/2)), axis=-1, arr=vertical)
    ffteast = np.apply_along_axis(lambda t: np.abs(rfft(t)/(N/2)), axis=-1, arr=east)

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


    return fftnorth, fftvertical, ffteast, freq


def plot_fft(north, vertical, east, freq, name: str, fmin: float, fmax: float):
    """Plots the FFT spectrum 

    Args
    ----
        - north (ndarray): North-component signal
        - vertical (ndarray): Vertical-component signal
        - east (ndarray): East-component signal
        - freq (ndarray): Frequency vector 
        - name (str): name of the file

    Changelog
    ---------
    - 08/SEP/2023:\n
        --> Corte del vector de frecuencias a los límites establecidos\n
        --> Corte de los vectores de datos a los límites establecidos
    - 09/SEP/2023: \n
        --> Removed the vector cropping function and moved it to the rhs_spectrum one
    """

    # Calcula el promedio de las ventanas
    n_mean = np.mean(north, axis=0)
    v_mean = np.mean(vertical, axis=0)
    e_mean = np.mean(east, axis=0)

    nombre = os.path.basename(name)[:4]

    plt.semilogx(freq, v_mean, color='r', label='Z', lw=1)
    plt.semilogx(freq, n_mean, color='g', label='N', lw=1)
    plt.semilogx(freq, e_mean, color='b', label='E', lw=1)

    
    plt.title(f'{nombre} - FFT', fontsize=15)
    plt.xlabel("Frequency [Hz]", fontsize=12, labelpad=10)
    plt.ylabel('Power Spectral Density [(counts)^2/Hz]', fontsize=12, labelpad=10)
    plt.xlim(fmin, fmax)
    plt.grid(ls='--', which='both')
    plt.legend()
    plt.tick_params('both', labelsize=12)

    # plt.savefig(f'{name[:-4]}-FFT.png', dpi=400)
    # plt.show()

    fig = plt.gcf()
    return fig


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


def konnoohmachi_smoothing(fftnorth, fftvertical, ffteast, freq, bandwidth):
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


def konnoohmachi_smoothing_opt(fftnorth, fftvertical, ffteast, freq, bandwidth):
    """Smooths the data using Konno-Ohmachi (1998) algorithm.\n
    Optimized version with numpy vectorization and cython for smoothing.\n
    Up to 2x faster than normal version when already allocated in memory

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
    
    Changelog
    ---------
    - 10/SEP/23:\n
        --> Added the function
    """
    s = perf_counter()

    north_smooth = np.apply_along_axis(
                    func1d=lambda t: pykooh.smooth(freq, freq, t, b=bandwidth, use_cython=True), 
                    arr=fftnorth, 
                    axis=-1
                    )
    vertical_smooth = np.apply_along_axis(
                    func1d=lambda t: pykooh.smooth(freq, freq, t, b=bandwidth, use_cython=True), 
                    arr=fftvertical, 
                    axis=-1
                    )
    east_smooth = np.apply_along_axis(
                    func1d=lambda t: pykooh.smooth(freq, freq, t, b=bandwidth, use_cython=True), 
                    arr=ffteast, 
                    axis=-1
                    )
    
    e = perf_counter()
    # print(f'{round(e-s, 4)} s transcurridos')

    return north_smooth, vertical_smooth, east_smooth


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


def hv_ratio(amp_north, amp_vertical, amp_east, freq, fmin=0.1, fmax=10):
    # TODO: cortar los vectores a fmin y fmax
    # 
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
                 indices_or_sections=[fmin_loc, fmax_loc], axis=-1)[1])

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


def plot_hv(HV_mean, HV, freq, fmin, fmax, name, plot_windows=True):
    """Generates a plot for the HV Spectral Ratio, indicating all the analysis windows and the position of the maximum amplitude 

    Args:
    ----
        - HV_mean (ndarray): 1D HV spectrum with the mean value of all the windows. If there's only one window, HV and HV_mean will be the same.
        - HV (ndarray): HV spectrum of all the windows. It can be a 2D array
        - freq (ndarray): Frequency array
        - fmin (float): Minimum frequency for the plot
        - fmax (float): Maximum frequency for the plot
        - name (str): Name of the analyzed seismic station or path of the original analyzed file. Title of the plot.
        - plot_windows (bool, optional): Whether to show all the analysis windows or not. If False, only the mean value is visible. Defaults to True.

    Returns:
    -------
        fig: matplotlib figure with the HV Spectral Ratio graph
    
    Changelog:
    ---------
        - 09/SEP/23:\n
            --> Changed xmin, xmax to fmin, fmax
            --> Added textbox to the upper right corner to indicate frequency and amplitude
    """    

    maxval = np.nanargmax(HV_mean)

    if os.path.basename(name).endswith('.txt') or '.' not in os.path.basename(name):
        nombre = os.path.basename(name)[:4]
    elif name is None:
        nombre = name
    else:
        nombre = os.path.basename(name.split(".")[0])


    if plot_windows==True:
        for count, i in enumerate(HV):
            # for i, count in zip(HV, range(len(HV))):
            plt.semilogx(freq, i, label=f'Ventana {count+1}', lw=0.5)

    plt.vlines(freq[maxval], 0, np.nanmax(HV_mean)*1.5,
               colors='#808080', linewidths=20, alpha=0.5)
    plt.vlines(freq[maxval], 0, np.nanmax(HV_mean)
               * 1.5, colors='#363636', linewidths=1)

    plt.semilogx(freq, HV_mean, color='r')

    xtext = round(freq[maxval], 4)
    ytext = round(np.nanmax(HV_mean), 4)
    plt.text(x=0.95, y=0.95, s=f'f = {xtext} Hz \n Amp = {ytext}', bbox=dict(facecolor='white', edgecolor='black', pad=5.0), transform=plt.gca().transAxes, ha='right', va='top')

    plt.xlabel("Frequency [Hz]", fontsize=12, labelpad=10)
    plt.ylabel('H/V', fontsize=12, labelpad=10)
    plt.xlim(fmin, fmax)
    plt.ylim(0, np.nanmax(HV_mean)*1.25)
    plt.title(f'{nombre} - H/V', fontsize=15)
    plt.grid(ls='--', which='both')
    plt.tick_params('both', labelsize=12)
    # plt.legend()

    # plt.savefig(f'{name[:-4]}-HV.png', dpi=400)
    # plt.show()

    fig = plt.gcf()
    return fig


def save_results(HV_mean, fftnorth, fftvertical, ffteast, freq, name, which='both'):
    """Save the analysis results to a TXT file

    Args:
        - HV_mean (ndarray): HV spectrum with the mean of all the windows
        - fftnorth (ndarray): FFT spectrum for the north component
        - fftvertical (ndarray): FFT spectrum for the vertical component
        - ffteast (ndarray): FFT spectrum for the east component
        - freq (ndarray): Frequency array for HV and FFT
        - name (str): original path of the analyzed file
        - which (str, optional): Whether to save FFT, HV or both. Defaults to 'both'.
    """   
    
    if '.' not in os.path.basename(name):
        filename = name + '-RESULTADOS.txt'
    else:
        filename = name[:-4] + '-RESULTADOS.txt'

    n_mean = np.mean(fftnorth, axis=0)
    v_mean = np.mean(fftvertical, axis=0)
    e_mean = np.mean(ffteast, axis=0)

    maxHV = round(np.nanmax(HV_mean), 4)
    maxHV_ind = np.nanargmax(HV_mean)
    freqmaxHV = round(freq[maxHV_ind], 4)

    if which == 'both':
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(f'** RESULTADOS H/V & FFT**\n\n')
            file.write(f'Amplitud H/V máxima: {maxHV} \nFrecuencia de amplitud H/V máxima: {freqmaxHV} Hz \nPeriodo de amplitud H/V máxima: {round(1/freqmaxHV, 4)} s\n\n')
            file.write(f'ID \t Frecuencia \t H/V \t FFTN \t FFTZ \t FFTE \n')
            for i, f, hv, fftn, fftz, ffte in zip(range(len(HV_mean)), freq, HV_mean, n_mean, v_mean, e_mean):
                file.write(str(i) + '\t' + str(f) + '\t' + str(hv) + '\t' + str(fftn) + '\t' + str(fftz) + '\t' + str(ffte) + '\n')
        # print('\nArchivo guardado!')

    elif which == 'fft':
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(f'** RESULTADOS FFT**\n\n')
            # file.write(f'Amplitud H/V máxima: {maxHV} \nFrecuencia de amplitud H/V máxima: {freqmaxHV} Hz \nPeriodo de amplitud H/V máxima: {round(1/freqmaxHV, 4)} s\n\n')
            file.write(f'ID \t Frecuencia \t FFTN \t FFTZ \t FFTE \n')
            for i, f, fftn, fftz, ffte in zip(range(len(n_mean)), freq, n_mean, v_mean, e_mean):
                file.write(str(i) + '\t' + str(f) + '\t' + str(fftn) + '\t' + str(fftz) + '\t' + str(ffte) + '\n')
        # print('\nArchivo guardado!')

    elif which == 'hv':
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(f'** RESULTADOS H/V & FFT**\n\n')
            file.write(f'Amplitud H/V máxima: {maxHV} \nFrecuencia de amplitud H/V máxima: {freqmaxHV} Hz \nPeriodo de amplitud H/V máxima: {round(1/freqmaxHV, 4)} s\n\n')
            file.write(f'ID \t Frecuencia \t H/V \n')
            for i, f, hv in zip(range(len(HV_mean)), freq, HV_mean):
                file.write(str(i) + '\t' + str(f) + '\t' + str(hv) + '\n')
        # print('\nArchivo guardado!')


def process_hvsr(data, dt, win_len, taper_win, smooth_band, fftmin, fftmax, hvmin, hvmax):
    """Calculates H/V Spectral Ratio from acceleration data, either from variable or file.


    Args:
    -----
        - data (NDArray or string): Path of the file that contains the data (str) or variable with data (NDArray)

        - dt (float): Sampling period of the signal

        - win_len (float): Length in seconds of the analysis window

        - taper_win (str): Window type for taper. Valid windows are those in the scipy.signal.windows module, but only the ones that does not take any special parameters: {'barthann','bartlett','blackman','blackmanharris','bohman','boxcar','chebwin','cosine','exponential','flattop','hamming','hann','lanczos','nuttall','parzen','taylor','triang','tukey'}

        - smooth_band (float): Smooth coefficient of the Konno-Ohmachi smoothing function.
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
    6. Smooth spectra
    7. Calculate H/V
    """
    fs = 1/dt

    # data_windowed returns a 3-element tuple (N, V, E) and window number
    # return [north, vertical, east], window_number
    data_windowed, win_num = window(win_len, data, fs)
    print('Data windowed')

    try:
        north = data_windowed[0]
        vertical = data_windowed[1]
        east = data_windowed[2]
    except IndexError as ie:
        traceback.print_exc()
        print('Windowed array has more than 3 sub-arrays')

    # data_cropped returns a 3-element tuple (N, V, E)
    # return north, vertical, east
    data_cropped = crop_signal(north, vertical, east)

    try:
        north = data_cropped[0]
        vertical = data_cropped[1]
        east = data_cropped[2]
    except IndexError as ie:
        traceback.print_exc()
        print('Cropped array has more than 3 sub-arrays')
    print('\nData cropped')
    

    # taper returns a 3-element tuple (N-tapered, V-tapered, E-tapered)
    data_tapered = taper(north, vertical, east, taper_win)

    try:
        north = data_tapered[0]
        vertical = data_tapered[1]
        east = data_tapered[2]
    except IndexError as ie:
        traceback.print_exc()
        print('Tapered array has more than 3 sub-arrays')
    print('\nData tapered')


    # rhs_spectrum returns a 4-element tuple (NFFT, VFFT, EFFT, Frequency)
    # return fftnorth, fftvertical, ffteast, freq
    data_fft = rhs_spectrum(north, vertical, east, dt, fmin=fftmin, fmax=fftmax)

    try:
        freq = data_fft[-1]
        north = data_fft[0]
        vertical = data_fft[1]
        east = data_fft[2]
    except IndexError as ie:
        traceback.print_exc()
        print('FFT array has more than 4 sub-arrays')
    print('\nFFT done')


    # konnoohmachi_smoothing_opt returns a 3-element tuple (N-smooth, V-smooth, E-smooth)
    print('\nSmoothing... this may take a while')
    data_smoothed = konnoohmachi_smoothing_opt(north, vertical, east, freq, smooth_band)

    try:
        north = data_smoothed[0]
        vertical = data_smoothed[1]
        east = data_smoothed[2]
    except IndexError as ie:
        traceback.print_exc()
        print('Smoothed array has more than 3 sub-arrays')
    print('\nSmoothing done')


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
    print('\n Spectral ratio calculated!')


    return HV_mean, HV, freq

def hvsr(acc_data, file_type, dt, win_len, taper_window='cosine', smooth_bandwidth=40.0, fftmin=0.1, fftmax=50.0, hvmin=0.1, hvmax=10.0):
    """Calculates H/V Spectral Ratio given the acceleration data wiht north, east and vertical components

    Args:
    -----
        - acc_data (ndarray, str): Path of the file that contains the data (str) or variable with data (NDArray)

        - file_type (str): File type. Valid options are {'cires','ascii','mseed','miniseed','sac'}

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

    
    if isinstance(acc_data, str):
        # checks if acc_data is path-like
        try:
            if file_type.lower() == 'cires':
                data = read_cires(acc_data)
            elif file_type.lower() == 'ascii':
                data = read_file(acc_data)
            elif file_type.lower() == 'mseed' or file_type.lower() == 'miniseed':
                data = read_mseed(acc_data)
            elif file_type.lower() == 'sac':
                data = read_sac(acc_data)
        except FileNotFoundError as fnf:
            traceback.print_exc()

        hvmean, hv, freq = process_hvsr(data=data, dt=dt, win_len=win_len, taper_win=taper_window, smooth_band=smooth_bandwidth, fftmin=fftmin, fftmax=fftmax, hvmin=hvmin, hvmax=hvmax)
    else:
        # if acc_data is a variable (override reading the file)
        hvmean, hv, freq = process_hvsr(data=acc_data, dt=dt, win_len=win_len, taper_win=taper_window, smooth_band=smooth_bandwidth, fftmin=fftmin, fftmax=fftmax, hvmin=hvmin, hvmax=hvmax)
    
    return hvmean, hv, freq
