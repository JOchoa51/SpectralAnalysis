import random
import matplotlib
matplotlib.use('Agg')
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
import scienceplots
import os
import datetime
import colorsys

plt.style.use(["science", "notebook", "grid"])


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

    # print('\nVentanas disponibles: ')
    # for i in windows.__all__:
    #     print(f'\t{i}')
    # type = input('\nTipo de ventana: ')

    while True:
        try:
            w = windows.get_window(type, len(north[0]))
            north = np.array([n*w for n in north])
            vertical = np.array([v*w for v in vertical])
            east = np.array([e*w for e in east])

            return north, vertical, east, w

        except ValueError as e:
            type = input('Ventana no válida. Intenta nuevamente: ')



def window(window_length: float, data, fs: float):
    """Split the signal into the given number of sections

    Args
    ----
        sections (float): Length of sections in seconds
        data (ndarray): Signal data

    Returns
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

    # colors = ['#FF0000', '#FF4500', '#FF6900', '#FF8C00', '#FFAF00', '#FFD200', '#FFFF00', '#FFFF00', '#FFFF00', '#FFFF00']
    
    # color list with random, vivid colors  in html code
    # colors = ['#'+ ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(10)]

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


def plot_signal_windowed(north, vertical, east, dt, window_type, name):
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


def rhs_spectrum(north, vertical, east, dt):
    """Calculates the right-hand-side amplitude spectrum of the signal

    Args
    ----
        north (ndarray): North-component signal
        vertical (ndarray): Vertical-component signal
        east (ndarray): East-component signal

    Returns
    -------
        tuple: Tuple that contains the north, vertical and east amplitude spectrums
    """
    n = len(north[0])  # longitud de cada seccion
    freq = rfftfreq(n, dt)
    fftnorth = [rfft(no)/(n/2) for no in north]
    ffteast = [rfft(e)/(n/2) for e in east]
    fftvertical = [rfft(v)/(n/2) for v in vertical]

    fftnorth = np.array([np.abs(i) for i in fftnorth])
    fftvertical = np.array([np.abs(j) for j in fftvertical])
    ffteast = np.array([np.abs(k) for k in ffteast])

    return fftnorth, fftvertical, ffteast, freq


def plot_fft(north, vertical, east, freq, name, xmin, xmax):
    """Plots the FFT spectrum 

        Args
        ----
            north (ndarray): North-component signal
            vertical (ndarray): Vertical-component signal
            east (ndarray): East-component signal
            freq (ndarray): Frequency vector 
            name (string): name of the file
    """

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
    plt.xlim(xmin, xmax)
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
        fftnorth (ndarray): North-component amplitude spectrum
        fftvertical (ndarray): Vertical-component amplitude spectrum
        ffteast (ndarray): East-component amplitude spectrum
        smoothie (int): Degree of smoothing

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
    """Smooths the data using Konno-Ohmachi (1981) algorithm

    Args
    ----
        fftnorth (ndarray): North-component amplitude spectrum
        fftvertical (ndarray): Vertical-component amplitude spectrum
        ffteast (ndarray): East-component amplitude spectrum
        freq (ndarray): Frequency vector
        bandwidth (int): Strength of the filter. A lower bandwidth is a stronger smoothing.

    Returns
    -------
        tuple: smoothed spectrum
    """
    north_smooth = []
    east_smooth = []
    vertical_smooth = []
    count = 0
    s = perf_counter()  # Cuenta el tiempo de ejecución del ciclo
    for i, j, k in zip(fftnorth, fftvertical, ffteast):
        count += 1

        # WITH PYKOOH
        north_smooth.append(pykooh.smooth(freq, freq, i, b=bandwidth))
        vertical_smooth.append(pykooh.smooth(freq, freq, j, b=bandwidth))
        east_smooth.append(pykooh.smooth(freq, freq, k, b=bandwidth))

        # print(f'Ventana {count} lista')

    # Convierte lista en numpy array
    north_smooth = np.array(north_smooth, dtype=object)
    vertical_smooth = np.array(vertical_smooth, dtype=object)
    east_smooth = np.array(east_smooth, dtype=object)

    e = perf_counter()
    # print(f'{round(e-s, 4)} s transcurridos')

    return north_smooth, vertical_smooth, east_smooth


def hv_ratio(amp_north, amp_vertical, amp_east):
    """Calculates the H/V spectral ratio

    Args
    ----
        amp_north (ndarray): North-component amplitude spectrum
        amp_vertical (ndarray): Vertical-component amplitude spectrum
        amp_east (ndarray): East-component amplitude spectrum

    Returns
    -------
        tuple: tuple containing the H/V mean ratio and the H/V for each window
    """
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
        HV_mean = list(np.mean(HV, axis=0))
    else:
        HV_mean = HV


    return HV_mean, HV

def save_results(HV_mean, fftnorth, fftvertical, ffteast, freq, name, which='both'):
    """
    Save the results of the analysis in a text file

    Args:
    ----
    HV_mean: array
        Mean value of every window of the HV
    fftnorth: array

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


    
        


def plot_hv(HV_mean, HV, freq, xmin, xmax, name, plot_windows=True):
    # xmin = float(input('Frecuencia minima de la gráfica: '))
    # xmax = float(input('Frecuencia máxima de la gráfica: '))
    # xmin = 0.1
    # xmax = 10
    maxval = np.nanargmax(HV_mean)

    if os.path.basename(name).endswith('.txt') or '.' not in os.path.basename(name):
        nombre = os.path.basename(name)[:4]
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

    plt.xlabel("Frequency [Hz]", fontsize=12, labelpad=10)
    plt.ylabel('H/V', fontsize=12, labelpad=10)
    plt.xlim(xmin, xmax)
    plt.ylim(0, np.nanmax(HV_mean)*1.25)
    plt.title(f'{nombre} - H/V', fontsize=15)
    plt.grid(ls='--', which='both')
    plt.tick_params('both', labelsize=12)
    # plt.legend()

    # plt.savefig(f'{name[:-4]}-HV.png', dpi=400)
    # plt.show()

    fig = plt.gcf()
    return fig

