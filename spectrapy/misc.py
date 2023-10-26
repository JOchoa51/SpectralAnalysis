# matplotlib.use('Agg')
# plt.rcParams['figure.dpi'] = 400
# plt.rcParams['figure.figsize'] = [10,10]

import numpy as np
import os


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

