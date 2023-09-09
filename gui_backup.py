import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGridLayout, QPushButton, QFileDialog, QTextEdit, QComboBox, QSplitter, QVBoxLayout, QHBoxLayout, QDoubleSpinBox, QCheckBox, QTabWidget, QDesktopWidget, QLCDNumber
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import processing as pr
import matplotlib.pyplot as plt

class SpectralAnalysisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Spectral Analysis')
        self.setGeometry(100, 100, 1000, 800)
        self.setup_ui()

        self.data = None
        self.file_path = None
        self.fs = None

    def setup_ui(self):
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        info_label = QLabel('Version 1.0 - © Chuy Inc., 2023')

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        main_layout.addWidget(info_label)

        tab_pre = QWidget()
        tab_params = QWidget()
        tab_fft = QWidget()
        tab_hv = QWidget() 

        # Create a new tab
        self.tab_widget.addTab(tab_pre, 'File')

        tab_pre_layout = QVBoxLayout(tab_pre)
        splitter = QSplitter()
        tab_pre_layout.addWidget(splitter)

        self.left_widget = QWidget()
        self.left_layout = QGridLayout()
        self.left_widget.setLayout(self.left_layout)

        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout()
        self.right_widget.setLayout(self.right_layout)

        splitter.addWidget(self.left_widget)
        splitter.addWidget(self.right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1,1])  # Equal sizes

        self.create_left_panel()
        self.create_plot_panel()

        self.tab_widget.addTab(tab_params, 'Processing parameters')
        self.params_layout(tab_params)
        # Disable the tab to prevent errors from handling empty data
        self.params_tab_index = 1
        self.tab_widget.widget(self.params_tab_index).setEnabled(False)
        

        self.tab_widget.addTab(tab_fft, 'FFT')
        self.fft_layout(tab_fft)
        self.fft_tab_index = 2
        self.tab_widget.widget(self.fft_tab_index).setEnabled(False)

        self.tab_widget.addTab(tab_hv, 'HV')
        self.hv_layout(tab_hv)
        self.hv_tab_index = 3
        self.tab_widget.widget(self.hv_tab_index).setEnabled(False)
    
    def enable_tab(self, tab_index):
        self.tab_widget.widget(tab_index).setEnabled(True)

    def create_left_panel(self):
        self.text_edit = QTextEdit()
        self.text_edit.setLineWrapMode(QTextEdit.WidgetWidth)
        self.text_edit.setReadOnly(True)

        self.button_open = QPushButton("Open File")
        self.button_open.setFixedWidth(self.text_edit.sizeHint().width())
        self.button_open.clicked.connect(self.open_file)

        self.options_combo = QComboBox()
        self.options_combo.addItem("Select file type...")
        self.options_combo.addItem("ASCII")
        self.options_combo.addItem("CIRES")
        self.options_combo.addItem("MSEED")
        self.options_combo.addItem("SAC")
        self.options_combo.currentIndexChanged.connect(self.file_type_changed)
        self.options_combo.setFixedWidth(self.text_edit.sizeHint().width())

        self.samplerate_label = QLabel('Samplerate: ')
        self.samplerate_spinbox = QDoubleSpinBox()
        self.samplerate_spinbox.setMaximum(99999)
        self.samplerate_spinbox.setMinimum(1)
        self.samplerate_spinbox.setSuffix(' Hz')

        self.samplerate_layout = QHBoxLayout()
        self.samplerate_layout.setAlignment(Qt.AlignLeft)
        self.samplerate_layout.addWidget(self.samplerate_label)
        self.samplerate_layout.addWidget(self.samplerate_spinbox)
        self.samplerate_widget = QWidget()
        self.samplerate_widget.setLayout(self.samplerate_layout)

        self.left_layout.addWidget(self.options_combo, 1, 0)
        self.left_layout.addWidget(self.button_open, 2, 0)
        self.left_layout.addWidget(self.samplerate_widget, 3, 0)
        self.left_layout.addWidget(self.text_edit, 4, 0)

        # Update the value in the spinbox
        self.samplerate_spinbox.valueChanged.connect(self.update_samplerate)

        # Disable the button initially
        self.button_open.setEnabled(False)

    def update_samplerate(self, samplerate_value):
        self.fs = samplerate_value
        self.display_signal()

    def create_plot_panel(self):
        self.plot_widget = QWidget()
        self.plot_layout = QVBoxLayout()
        self.plot_widget.setLayout(self.plot_layout)
        # self.plot_widget.setFixedWidth()

        self.right_layout.addWidget(self.plot_widget)

    def file_type_changed(self, index):
        # Enable the "Open File" button only if a different option is selected
        self.button_open.setEnabled(index != 0)

    def open_file(self):
        file_dialog = QFileDialog()
        self.file_path, _ = file_dialog.getOpenFileName(self, f'Select {self.options_combo.currentText()} File')

        if self.file_path:
            if self.file_path.endswith('.txt'):
                self.data = pr.read_file(self.file_path)
            elif self.file_path.endswith('.mseed') or self.file_path.endswith('.miniseed'):
                self.data = pr.read_mseed(self.file_path)
            elif self.file_path.endswith('.sac'):
                self.data = pr.read_sac(self.file_path)
            elif '.' not in os.path.basename(self.file_path):
                self.data = pr.read_cires(self.file_path)
            self.write_to_textbox(self.data)
        else:
            file_dialog.close()

        # THE DATA FOR THE FFT AND CONSEQUENT ANALYSIS COULD FIT HERE, BUT DOES IT?
    
    def write_to_textbox(self, data):
        file_type = self.options_combo.currentText()

        try:
            if file_type == 'CIRES':
                text = ''.join(data[3])
            else:
                text = 'N \t V \t E \n'
                for values in zip(data[0], data[1], data[2]):
                    text += '\t'.join(str(round(v, 4)) for v in values) + '\n'

            self.text_edit.setText(text)
            self.text_edit.setMinimumWidth(self.text_edit.sizeHint().width())
        except Exception as e:
            print(e)

        self.button_open.setText(f'File: {os.path.basename(self.file_path)}')

        self.display_signal()

    def display_signal(self):
        try:
            north = self.data[0]
            vertical = self.data[1]
            east = self.data[2]
        except TypeError:
            print('select a valid file')

        def plot_signal():
            dt = 1 / float(self.fs)
            figure = plt.clf()
            figure = pr.plot_signal(north, vertical, east, dt, self.file_path)
            canvas = FigureCanvas(figure)
            toolbar = NavigationToolbar(canvas)

            while self.plot_layout.count():
                item = self.plot_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

            self.plot_layout.addWidget(toolbar)
            self.plot_layout.addWidget(canvas)

            fig = plt.gcf()
            plt.close(fig)

        if self.fs is None:
            self.fs = 100

        plot_signal()
        self.samplerate_spinbox.setValue(self.fs)
        self.enable_tab(self.params_tab_index)

    def params_layout(self, tab):
        """
        Layout of the 'parameters' tab.

        It should contain (a QGridLayout):
            - A spinbox to select the window length
            - A combo box to select the taper window (add an icon to each one?)
            - A spinbox to select the smoothing constant (default to 40)
            - Two spinbox for min and max limits of the plot
            - A checkbox to show/not show windows in the HV plot
            - A plot layout for the windows on the bottom
                - The window length plot should change with the value
        """
        self.params_main_layout = QHBoxLayout(tab)

        self.params_submain_layout = QVBoxLayout()
        self.params_submain_layout.setAlignment(Qt.AlignVCenter)
        self.params_submain_widget = QWidget()
        self.params_submain_widget.setLayout(self.params_submain_layout)

        
        self.window_label = QLabel('Window length:')

        self.window_number_label = QLabel('Window number: ')

        self.win_num_lcd = QLCDNumber()
        self.win_num_lcd.setFixedSize(60, 25)
        self.win_num_lcd.setSegmentStyle(QLCDNumber.Flat)
        self.win_num_lcd.setDigitCount(3)
        self.win_num_lcd.setMode(QLCDNumber.Dec)

        self.window_length_spinbox = QDoubleSpinBox()
        self.window_length_spinbox.setSuffix(' s')
        self.window_length_spinbox.setMaximum(99999)
        self.window_length_spinbox.setMinimum(1)
        self.window_length_spinbox.setDecimals(2)
        # self.window_length_spinbox.setValue(81.00)

        self.taper_checkbox = QCheckBox()
        self.taper_checkbox.setFixedSize(20,20)

        self.taper_label = QLabel('Taper: ')

        self.taper_list = QComboBox()
        self.taper_list.setEditable(True)
        available_windows = ['boxcar', 'triang', 'parzen', 'bohman', 'blackman', 'nuttall', 'blackmanharris', 'flattop', 'bartlett', 'barthann', 'hamming', 'cosine', 'hann', 'exponential', 'tukey', 'taylor', 'lanczos']
        available_windows = [win.capitalize() for win in available_windows]
        
        self.taper_list.addItem('Select...')
        self.taper_list.addItems(available_windows)
        self.taper_list.setCurrentIndex(0)
        self.taper_list.setDisabled(True)
        self.taper_list.lineEdit().setReadOnly(True)

        try:
            icon_path = 'Icons\Windows'
            for win in enumerate(available_windows):
                index = win[0]
                for ic in os.listdir(icon_path):
                    if ic.split(".")[0] == win[1].lower():
                        image_path = os.path.join(icon_path, ic)
                        icon = QIcon(image_path)
                        self.taper_list.setItemIcon(index+1, icon)
        except FileNotFoundError:
            print('Icon not found')

        self.smooth_label = QLabel('Konno-Ohmachi \nsmoothing constant: ')
        self.smooth_spinbox = QDoubleSpinBox()
        self.smooth_spinbox.setMinimum(0)
        self.smooth_spinbox.setMaximum(100)
        self.smooth_spinbox.setSuffix(' Hz')
        self.smooth_spinbox.setValue(40)
        self.smooth_constant = self.smooth_spinbox.value()

        self.plot_button = QPushButton('Plot')  
        self.plot_button.setFixedHeight(30)
        self.plot_button.setFixedWidth(self.plot_button.sizeHint().width())

        # Window lenght QLabel and QDoubleSpinBox
        self.params_win_len_hlayout = QHBoxLayout()
        self.params_win_len_hlayout.setAlignment(Qt.AlignLeft)
        self.params_win_len_widget = QWidget()
        self.params_win_len_hlayout.addWidget(self.window_label)
        self.params_win_len_hlayout.addWidget(self.window_length_spinbox)
        self.params_win_len_widget.setLayout(self.params_win_len_hlayout)

        # Window number QLabel and QTextEdit
        self.params_win_num_hlayout = QHBoxLayout()
        self.params_win_num_hlayout.setAlignment(Qt.AlignLeft)
        self.params_win_num_widget = QWidget()
        self.params_win_num_hlayout.addWidget(self.window_number_label)
        self.params_win_num_hlayout.addWidget(self.win_num_lcd)
        self.params_win_num_widget.setLayout(self.params_win_num_hlayout)

        # Taper QCheckBox, QLabel and QComboBox
        self.params_taper_hlayout = QHBoxLayout()
        self.params_taper_hlayout.setAlignment(Qt.AlignLeft)
        self.params_taper_widget = QWidget()
        self.params_taper_hlayout.addWidget(self.taper_checkbox)
        self.params_taper_hlayout.addWidget(self.taper_label)
        self.params_taper_hlayout.addWidget(self.taper_list)
        self.params_taper_widget.setLayout(self.params_taper_hlayout)

        # Smoothing constant QLabel and QDoubleSpinBox
        self.params_smooth_hlayout = QHBoxLayout()
        self.params_smooth_hlayout.setAlignment(Qt.AlignLeft)
        self.params_smooth_widget = QWidget()
        self.params_smooth_hlayout.addWidget(self.smooth_label)
        self.params_smooth_hlayout.addWidget(self.smooth_spinbox)
        self.params_smooth_widget.setLayout(self.params_smooth_hlayout)

        self.params_submain_layout.addWidget(self.params_win_len_widget)
        self.params_submain_layout.addWidget(self.params_win_num_widget)
        self.params_submain_layout.addWidget(self.params_smooth_widget)
        self.params_submain_layout.addWidget(self.params_taper_widget)
        self.params_submain_layout.addWidget(self.plot_button)
        self.params_main_layout.addWidget(self.params_submain_widget)
        # self.params_main_layout.setAlignment(Qt.AlignTop)

        # Layout and widget for the plot
        self.params_box_plot_layout = QVBoxLayout()
        self.params_box_plot_widget = QWidget()
        self.params_box_plot_widget.setLayout(self.params_box_plot_layout)

        self.params_main_layout.addWidget(self.params_box_plot_widget)

        # Update the value in the window length spinbox
        self.window_length_spinbox.valueChanged.connect(self.update_window_length)

        # Update the value in the smoothing spinbox
        self.smooth_spinbox.valueChanged.connect(self.update_smooth_constant)


        # Enable taper_list upon checking checkBox
        self.taper_checkbox.stateChanged.connect(self.set_combobox_enabled)
        self.taper_list.currentIndexChanged.connect(self.update_selected_taper)

        # plot button
        self.plot_button.clicked.connect(self.plot_button_clicked)

    def plot_button_clicked(self):
        # Action taken when the Plot button is pressed
        if self.fs is not None:
            try:
                self.display_box_plot_windows(self.window_length)
            except AttributeError:
                self.window_length = 1
                self.display_box_plot_windows(self.window_length)
        

    def update_selected_taper(self):
        # gets the value in the combo box and converts it to lowercase so it matches the scipy function
        self.selected_taper = self.taper_list.currentText().lower()

        # self.data, self.win_type = pr.taper(self.data[0], self.data[1], self.data[2], self.selected_taper)

        # self.display_win_plot(self.win_type)

    def set_combobox_enabled(self):
        # Enables/disables the combobox based on the state of the checkbox
        self.taper_list.setEnabled(self.taper_checkbox.isChecked())


    def update_window_length(self, length_value):
        # Reads changes in the spinbox value
        self.window_length = length_value

        # Splits the data into the window number specified
        self.data_split, win_num = pr.window(length_value, self.data, self.fs)
        self.data_split = pr.crop_signal(self.data_split[0], self.data_split[1], self.data_split[2])

        # Update the value displayed in the QLCDNumber
        self.win_num_lcd.display(int(win_num))

    def update_smooth_constant(self, smooth_value):
        # Reads the smoothing constant = if it changes
        self.smooth_constant = smooth_value

    def display_box_plot_windows(self, length):
        dt = 1 / self.fs
        figure = plt.clf()
        figure = pr.plot_windows(self.data, length, dt, self.file_path)
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas)

        while self.params_box_plot_layout.count():
            item = self.params_box_plot_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.params_box_plot_layout.addWidget(toolbar)
        self.params_box_plot_layout.addWidget(canvas)

        fig = plt.gcf()
        plt.close(fig)    
    
        self.plot_button.setText('Update plot')
        self.enable_tab(self.fft_tab_index)
        self.enable_tab(self.hv_tab_index)


    def fft_layout(self, tab):
        # Set the layout for the FFT Tab

        self.fft_main_layout = QHBoxLayout(tab)

        self.fft_submain_layout = QVBoxLayout()
        self.fft_submain_layout.setAlignment(Qt.AlignVCenter)
        self.fft_submain_widget = QWidget()
        self.fft_submain_widget.setLayout(self.fft_submain_layout)
        
        # HBox layout that contains minimum values for FFT plot
        self.fft_from_hlayout = QHBoxLayout()
        # self.fft_from_hlayout.setAlignment(Qt.AlignLeft|Qt.AlignTop)
        self.fft_from_widget = QWidget()
        self.fft_minval_label = QLabel('From: ')
        self.fft_minval_spinbox = QDoubleSpinBox()
        self.fft_minval_spinbox.setMinimum(0.01)
        self.fft_minval_spinbox.setMaximum(100)
        self.fft_minval_spinbox.setValue(0.1)
        self.fft_minval_spinbox.setSuffix(' Hz')
        self.fft_from_hlayout.addWidget(self.fft_minval_label)
        self.fft_from_hlayout.addWidget(self.fft_minval_spinbox)
        self.fft_from_widget.setLayout(self.fft_from_hlayout)

        # HBox layout that contains maximum values for FFT plot
        self.fft_to_hlayout = QHBoxLayout()
        self.fft_to_hlayout.setAlignment(Qt.AlignLeft|Qt.AlignTop)
        self.fft_to_widget = QWidget()
        self.fft_maxval_label = QLabel('To: ')
        self.fft_maxval_spinbox = QDoubleSpinBox()
        self.fft_maxval_spinbox.setMinimum(0.1)
        self.fft_maxval_spinbox.setMaximum(100)
        self.fft_maxval_spinbox.setValue(50)
        self.fft_maxval_spinbox.setSuffix(' Hz')
        self.fft_to_hlayout.addWidget(self.fft_maxval_label)
        self.fft_to_hlayout.addWidget(self.fft_maxval_spinbox)
        self.fft_to_widget.setLayout(self.fft_to_hlayout)

        # Plot layout for FFT plot
        self.fft_plot_layout = QVBoxLayout()
        self.fft_plot_widget = QWidget()
        self.fft_plot_widget.setLayout(self.fft_plot_layout)
        # self.fft_plot_layout.setAlignment(Qt.AlignCenter)

        # Button for for plotting FFT
        self.fft_plot_button = QPushButton('Plot')
        self.fft_plot_button.setFixedWidth(self.fft_plot_button.sizeHint().width())
        self.fft_plot_button.clicked.connect(self.fft_button_clicked)




        self.fft_submain_layout.addWidget(self.fft_from_widget)
        self.fft_submain_layout.addWidget(self.fft_to_widget)
        self.fft_submain_layout.addWidget(self.fft_plot_button)
        # self.fft_submain_layout.addWidget(self.fft_clear_canvas_button)
        self.fft_main_layout.addWidget(self.fft_submain_widget)
        self.fft_main_layout.addWidget(self.fft_plot_widget)
        # self.fft_main_layout.setAlignment(Qt.AlignTop|Qt.AlignLeft)

    def fft_button_clicked(self):
        # Action when plot fft button is clicked
        if self.fs is not None:
            self.display_fft()

    def remove_nohv_label(self):
        self.no_hv.setParent(None)

    def save_res(self):
        try:
            pr.save_results(self.hv_mean, self.north, self.vertical, self.east, self.freq, self.file_path, which='both')
            # self.no_hv.setParent(None) 
            self.hv_submain_layout.addWidget(self.file_saved)
        except AttributeError as e:
            print(e)


    def display_fft(self):
        """
            Changelog:
            --------- 
            - 09/SEP/23: \n
            Changed xmin, xmax to fmin, fmax. \n
            Moved fmin,fmax to the top in order to access them wherever in the function.\n
            Added fmin, fmax parameters to the pr.rhs_spectrum() in order to account for the changes made to said function.
            """
        # plot the FFT in the canvas
        self.fmin = self.fft_minval_spinbox.value()
        self.fmax = self.fft_maxval_spinbox.value()

        def plot(north, vertical, east):
            # plot FFT function just for the sake of saving space
            self.north, self.vertical, self.east = pr.konnoohmachi_smoothing(north, vertical, east, self.freq, self.smooth_constant)
            figure = plt.clf()
            figure = pr.plot_fft(self.north, self.vertical, self.east, self.freq, self.file_path, self.fmin, self.fmax)
            canvas = FigureCanvas(figure)
            toolbar = NavigationToolbar(canvas)

            while self.fft_plot_layout.count():
                item = self.fft_plot_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

            self.fft_plot_layout.addWidget(toolbar)
            self.fft_plot_layout.addWidget(canvas)
            
            fig = plt.gcf()
            plt.close(fig)

            self.fft_plot_button.setText('Update plot')

        if self.taper_checkbox.isChecked():
            north, vertical, east, _ = pr.taper(self.data_split[0], self.data_split[1], self.data_split[2], self.selected_taper)
            north, vertical, east, self.freq = pr.rhs_spectrum(self.data_split[0], self.data_split[1], self.data_split[2], 1/self.fs, self.fmin, self.fmax)
            plot(north, vertical, east)
        else:
            north, vertical, east, self.freq = pr.rhs_spectrum(self.data_split[0], self.data_split[1], self.data_split[2], 1/self.fs, self.fmin, self.fmax)
            plot(north, vertical, east)   

    def hv_layout(self, tab):
        # The HV layout is pretty much a copy of the FFT, but with its corresponding tweaks

        self.hv_main_layout = QHBoxLayout(tab)

        self.hv_submain_layout = QVBoxLayout()
        self.hv_submain_layout.setAlignment(Qt.AlignVCenter)
        self.hv_submain_widget = QWidget()
        self.hv_submain_widget.setLayout(self.hv_submain_layout)
        
        # HBox layout that contains minimum values for hv plot
        self.hv_from_hlayout = QHBoxLayout()
        # self.hv_from_hlayout.setAlignment(Qt.AlignLeft|Qt.AlignTop)
        self.hv_from_widget = QWidget()
        self.hv_minval_label = QLabel('From: ')
        self.hv_minval_spinbox = QDoubleSpinBox()
        self.hv_minval_spinbox.setMinimum(0.01)
        self.hv_minval_spinbox.setMaximum(100)
        self.hv_minval_spinbox.setValue(0.1)
        self.hv_minval_spinbox.setSuffix(' Hz')
        self.hv_from_hlayout.addWidget(self.hv_minval_label)
        self.hv_from_hlayout.addWidget(self.hv_minval_spinbox)
        self.hv_from_widget.setLayout(self.hv_from_hlayout)

        # HBox layout that contains maximum values for hv plot
        self.hv_to_hlayout = QHBoxLayout()
        # self.hv_to_hlayout.setAlignment(Qt.AlignLeft|Qt.AlignTop)
        self.hv_to_widget = QWidget()
        self.hv_maxval_label = QLabel('To: ')
        self.hv_maxval_spinbox = QDoubleSpinBox()
        self.hv_maxval_spinbox.setMinimum(0.1)
        self.hv_maxval_spinbox.setMaximum(100)
        self.hv_maxval_spinbox.setValue(10)
        self.hv_maxval_spinbox.setSuffix(' Hz')
        self.hv_to_hlayout.addWidget(self.hv_maxval_label)
        self.hv_to_hlayout.addWidget(self.hv_maxval_spinbox)
        self.hv_to_widget.setLayout(self.hv_to_hlayout)

        # Checkbox to decide wether the plot contains all the windows or just the mean value
        self.hv_windows_layout = QHBoxLayout()
        self.hv_windows_widget = QWidget()
        self.hv_windows_label = QLabel('Show all windows')
        self.hv_windows_checkbox = QCheckBox()
        self.hv_windows_layout.addWidget(self.hv_windows_checkbox)
        self.hv_windows_layout.addWidget(self.hv_windows_label)
        self.hv_windows_widget.setLayout(self.hv_windows_layout)
        
        # Plot layout for hv plot
        self.hv_plot_layout = QVBoxLayout()
        self.hv_plot_widget = QWidget()
        self.hv_plot_widget.setLayout(self.hv_plot_layout)
        # self.hv_plot_layout.setAlignment(Qt.AlignCenter)

        # Button for for plotting hv
        self.hv_plot_button = QPushButton('Plot')
        self.hv_plot_button.setFixedWidth(self.hv_plot_button.sizeHint().width())
        self.hv_plot_button.clicked.connect(self.hv_button_clicked)

        # Button to save results
        self.save_res_button = QPushButton('Save results')
        self.save_res_button.setFixedWidth(self.save_res_button.sizeHint().width())
        self.save_res_button.clicked.connect(self.save_res)
        
        self.file_saved = QLabel('File Saved!')
        self.file_saved.setAlignment(Qt.AlignCenter)

        # self.hv_clear_canvas_button.clicked.connect(self.clear_canvas)

        self.hv_submain_layout.addWidget(self.hv_from_widget)
        self.hv_submain_layout.addWidget(self.hv_to_widget)
        self.hv_submain_layout.addWidget(self.hv_windows_widget)
        self.hv_submain_layout.addWidget(self.hv_plot_button)
        self.hv_submain_layout.addWidget(self.save_res_button)
        self.hv_main_layout.addWidget(self.hv_submain_widget)
        self.hv_main_layout.addWidget(self.hv_plot_widget)
        # self.hv_main_layout.setAlignment(Qt.AlignTop|Qt.AlignLeft)

    def hv_button_clicked(self):
        # Action when plot hv button is clicked
        if self.fs is not None:
            self.display_hv()


    def display_hv(self):
        """
        Changelog
        ---------
        09/SEP/23:\n
            --> Added freq, fmin and fmax parameters to pr.hv_ratio()\n
            --> Added hv_freq as a variable for the pr.hv_ratio(), as it is required for the pr.plot_hv() function
        """
        # plot the hv in the canvas
        fmin = self.hv_minval_spinbox.value()
        fmax = self.hv_maxval_spinbox.value()
        self.hv_mean, hv_windows, hv_freq = pr.hv_ratio(self.north, self.vertical, self.east, self.freq, fmin, fmax)
        figure = plt.clf()
        if self.hv_windows_checkbox.isChecked():
            figure = pr.plot_hv(self.hv_mean, hv_windows, hv_freq, fmin, fmax, self.file_path, plot_windows=True)
        else:
            figure = pr.plot_hv(self.hv_mean, hv_windows, hv_freq, fmin, fmax, self.file_path, plot_windows=False)
        
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas)

        while self.hv_plot_layout.count():
            item = self.hv_plot_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.hv_plot_layout.addWidget(toolbar)
        self.hv_plot_layout.addWidget(canvas)
        
        fig = plt.gcf()
        plt.close(fig)

        self.hv_plot_button.setText('Update plot')
        

        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SpectralAnalysisApp()
    window.show()

    # Center the window on the screen
    window_rect = window.geometry()
    center_point = QDesktopWidget().availableGeometry().center()
    window_rect.moveCenter(center_point)
    window.setGeometry(window_rect)

    sys.exit(app.exec_())
