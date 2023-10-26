# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt.rcParams['figure.dpi'] = 400
# plt.rcParams['figure.figsize'] = [10,10]
from matplotlib.patches import Rectangle

import numpy as np


import os
import colorsys

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

    maxvalue = np.maximum(np.abs(north), np.abs(east))
    maxvalue = np.maximum(maxvalue, np.abs(vertical))
    maxvalue = np.max(maxvalue)*1.2
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
    # ax.set_yticks([])

    ax = axes[1]
    ax.plot(time, vertical, lw=0.5, color='k')
    ax.fill_between(time, vertical, where=(vertical > 0), color='black')
    # ax.set_title('Vertical', fontsize=12)
    ax.set_ylim(minvalue, maxvalue)
    ax.set_ylabel(f'{nombre} Z', rotation=0, labelpad=30)
    # ax.set_yticks([])

    ax = axes[2]
    ax.plot(time, east, lw=0.5, color='k')
    ax.fill_between(time, east, where=(east > 0), color='black')
    # ax.set_title('East', fontsize=12)
    ax.set_ylim(minvalue, maxvalue)
    ax.set_ylabel(f'{nombre} E', rotation=0, labelpad=30)
    # ax.set_yticks([])

    fig.supxlabel('Time [s]', fontsize=12)
    plt.tick_params('x', labelsize=12)
     
    plt.subplots_adjust(hspace=0.5)
    
    # # plt.show()

    return fig

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

def plot_signal_windows(north, vertical, east, dt, name, window_type='cosine'):
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
        --> Removed the vector cropping function and moved it to the spectrum one
    """

    # Calcula el promedio de las ventanas
    north = np.array(north)
    vertical = np.array(vertical)
    east = np.array(east)
    if north.ndim != 1 and vertical.ndim != 1 and east.ndim != 1:
        n_mean = np.mean(north, axis=0)
        v_mean = np.mean(vertical, axis=0)
        e_mean = np.mean(east, axis=0)
    else:
        n_mean = north
        v_mean = vertical
        e_mean = east

    nombre = os.path.basename(name)[:4]

    plt.semilogx(freq, v_mean, color='r', label='Z', lw=1)
    plt.semilogx(freq, n_mean, color='g', label='N', lw=1)
    plt.semilogx(freq, e_mean, color='b', label='E', lw=1)

    
    plt.title(f'{nombre} - FFT', fontsize=15)
    plt.xlabel("Frequency [Hz]", fontsize=12, labelpad=10)
    plt.ylabel('Amplitude', fontsize=12, labelpad=10)
    plt.xlim(fmin, fmax)
    plt.grid(ls='--', which='both')
    plt.legend()
    plt.tick_params('both', labelsize=12)

    # plt.savefig(f'{name[:-4]}-FFT.png', dpi=400)
    # plt.show()

    fig = plt.gcf()
    return fig

def plot_hv(HV_mean, HV, freq, fmin, fmax, name=None, plot_windows=True, period_or_freq='freq'):
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
        - period_or_freq (str, optional): Whether to plot HVSR against period or frequency. Defaults to 'freq'.

    Returns:
    -------
        fig: matplotlib figure with the HV Spectral Ratio graph
    
    Changelog:
    ---------
        - 09/SEP/23:\n
            --> Changed xmin, xmax to fmin, fmax
            --> Added textbox to the upper right corner to indicate frequency and amplitude
        - 14/SEP/23:\n
            --> Added the option to plot against frequency or period
    """    

    maxval = np.nanargmax(HV_mean)

    if name is None:
        nombre = ''
    elif name is not None:
        nombre = name
    elif os.path.basename(name).endswith('.txt') or '.' not in os.path.basename(name):
        nombre = os.path.basename(name)[:4]
    else:
        nombre = os.path.basename(name.split(".")[0])

    ytext = round(np.nanmax(HV_mean), 4)

    if period_or_freq == 'period':
        period = 1/freq

        if plot_windows==True and HV.ndim != 1:
            for count, i in enumerate(HV):
                # for i, count in zip(HV, range(len(HV))):
                plt.semilogx(period, i, label=f'Ventana {count+1}', lw=0.5, zorder=10)

        # period_mean = 1/HV_mean
        # plt.vlines(period[maxval], 0, np.nanmax(HV_mean)*1.5,
        #         colors='#808080', linewidths=20, alpha=0.5, zorder=5)
        # plt.vlines(period[maxval], 0, np.nanmax(HV_mean)
        #         * 1.5, colors='#363636', linewidths=1, zorder=5)
        plt.semilogx(period, HV_mean, color='r', zorder=10)

        xtext = round(period[maxval], 4)
        xtext_coord = 0.275
        plot_text = f'T = {xtext} s \n Amp = {ytext}'
        plt.xlabel("Period [s]", fontsize=12, labelpad=10)
    else:
        if plot_windows==True and HV.ndim != 1:
            for count, i in enumerate(HV):
                # for i, count in zip(HV, range(len(HV))):
                plt.semilogx(freq, i, label=f'Ventana {count+1}', lw=0.5, zorder=10)
        # plt.vlines(freq[maxval], 0, np.nanmax(HV_mean)*1.5,
        #         colors='#808080', linewidths=20, alpha=0.5, zorder=5)
        # plt.vlines(freq[maxval], 0, np.nanmax(HV_mean)
        #         * 1.5, colors='#363636', linewidths=1, zorder=5)
        plt.semilogx(freq, HV_mean, color='r', zorder=10)

        xtext = round(freq[maxval], 4)
        xtext_coord = 0.95
        plot_text = f'f = {xtext} Hz \n Amp = {ytext}'
        plt.xlabel("Frequency [Hz]", fontsize=12, labelpad=10)


    plt.text(x=xtext_coord, y=0.95, s=plot_text, bbox=dict(facecolor='white', edgecolor='black', pad=5.0), transform=plt.gca().transAxes, ha='right', va='top')

    plt.ylabel('H/V', fontsize=12, labelpad=10)
    plt.xlim(fmin, fmax)
    plt.ylim(0, np.nanmax(HV_mean)*1.25)
    plt.title(f'{nombre} - H/V', fontsize=15)
    plt.grid(ls='--', which='both', zorder=0)
    plt.tick_params('both', labelsize=12)
    # plt.legend()

    # plt.savefig(f'{name[:-4]}-HV.png', dpi=400)
    # plt.show()

    fig = plt.gcf()
    return fig
