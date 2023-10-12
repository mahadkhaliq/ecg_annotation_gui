# gui for performing annotation of the segments
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QComboBox, QPushButton, QHBoxLayout, QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtWidgets import QSizePolicy
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QGroupBox, QLabel, QSpacerItem
from PyQt5.QtWidgets import QGroupBox, QHBoxLayout, QVBoxLayout, QPushButton, QComboBox, QCheckBox, QWidget
from PyQt5.QtWidgets import QGridLayout


# Placeholder for ECG signal and time
ecg_signal = np.array([])
time_ms = np.array([])

# List to store segments
segments = []
# List to store segments and labels
all_segments = []
all_labels = []

r_peak_points = []
r_peak_labels = []

wave_segments = []
wave_labels = []
heartbeat_segments = []
heartbeat_labels = []

class App(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Interactive Labeling with PyQt5 and Matplotlib")
        self.setGeometry(100, 100, 800, 600)  # Increased window size slightly for better layout

        self.current_option="Please Select annotation type"

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # -- File and Basic Controls --
        file_group = QGroupBox("File Operations")
        file_layout = QHBoxLayout(file_group)

        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse)
        self.browse_button.setFixedWidth(150)
        self.browse_button.setFixedHeight(30)
        self.browse_button.setStyleSheet("background-color: dark_blue; color: White;")
        file_layout.addWidget(self.browse_button)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start)
        self.start_button.setFixedWidth(150)
        self.start_button.setFixedHeight(30)
        self.start_button.setStyleSheet("background-color: green; color: White;")
        file_layout.addWidget(self.start_button)


        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save)
        self.save_button.setFixedWidth(150)
        self.save_button.setFixedHeight(30)
        self.save_button.setStyleSheet("background-color: blue; color: White;")
        file_layout.addWidget(self.save_button)

        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(self.load)
        self.load_button.setFixedWidth(150)
        self.load_button.setFixedHeight(30)
        self.load_button.setStyleSheet('background-color: orange; color: White')
        file_layout.addWidget(self.load_button)



        self.close_button = QPushButton("Close Plot")
        self.close_button.clicked.connect(self.close_plot)
        self.close_button.setFixedWidth(150)
        self.close_button.setFixedHeight(30)
        self.close_button.setStyleSheet('background-color: brown; color: White')
        file_layout.addWidget(self.close_button)


        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.exit_app)
        self.exit_button.setFixedWidth(150)
        self.exit_button.setFixedHeight(30)
        self.exit_button.setStyleSheet('background-color: red; color: White')
        file_layout.addWidget(self.exit_button)

        main_layout.addWidget(file_group)

        # -- Navigation and Plot Controls --
        nav_group = QGroupBox("Plotting")
        nav_layout = QHBoxLayout(nav_group)

        # self.prev_button = QPushButton("Previous")
        # self.prev_button.clicked.connect(self.prev_plot)
        # nav_layout.addWidget(self.prev_button)
        #
        # self.next_button = QPushButton("Next")
        # self.next_button.clicked.connect(self.next_plot)
        # nav_layout.addWidget(self.next_button)
        #
        # self.start_button = QPushButton("Start")
        # self.start_button.clicked.connect(self.start)
        # nav_layout.addWidget(self.start_button)
        #
        # self.close_button = QPushButton("Close Plot")
        # self.close_button.clicked.connect(self.close_plot)
        # nav_layout.addWidget(self.close_button)
        #
        # self.exit_button = QPushButton("Exit")
        # self.exit_button.clicked.connect(self.exit_app)
        # nav_layout.addWidget(self.exit_button)
        #
        # self.interval_label = QLabel("Interval:")
        # nav_layout.addWidget(self.interval_label)
        #
        # self.interval_dropdown = QComboBox()
        # self.interval_dropdown.addItems([str(x) for x in range(200, 5100, 200)])
        # nav_layout.addWidget(self.interval_dropdown)
        #
        # main_layout.addWidget(nav_group)

        # -- Matplotlib Canvas and Navigation Toolbar --
        self.canvas = MplCanvas(self, width=100, height=100, dpi=100)
        main_layout.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        main_layout.addWidget(self.toolbar)

        status_group = QGroupBox("Status")
        status_layout = QHBoxLayout(status_group)
        # text = QLabel(self.current_option)
        # text.show()
        # status_layout.addWidget(text)
        main_layout.addWidget(status_group)

        self.interval = 0

        self.status_label = QLabel("ECG Status: Unfiltered | " + self.current_option + " | " + "Current Interval: " + str(self.interval) + "milliseconds", status_group)
        status_layout.addWidget(self.status_label)



        move_group = QGroupBox("Navigation")
        move_layout = QHBoxLayout(move_group)

        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.prev_plot)
        self.prev_button.setFixedWidth(80)  # set width to 120 pixels
        self.prev_button.setFixedHeight(30)
        self.prev_button.setStyleSheet("background-color: green; color: Black;")  # Set the background to blue and text to white
        move_layout.addWidget(self.prev_button)

        self.interval_label = QLabel("Interval:")
        self.interval_label.setFixedWidth(60)
        self.interval_label.setFixedHeight(30)
        move_layout.addWidget(self.interval_label)

        self.interval_dropdown = QComboBox()
        self.interval_dropdown.addItems([str(x) for x in range(200, 5100, 200)])
        self.interval_dropdown.setFixedWidth(150)
        self.interval_dropdown.setFixedHeight(30)
        move_layout.addWidget(self.interval_dropdown)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_plot)
        self.next_button.setFixedWidth(80)
        self.next_button.setFixedHeight(30)
        self.next_button.setStyleSheet("background-color: Red; color: Black;")  # Set the background to blue and text to white

        move_layout.addWidget(self.next_button)
        main_layout.addWidget(move_group)

        # -- Annotation Controls --
        anno_group = QGroupBox("Annotations and Filter")
        anno_layout = QGridLayout(anno_group)  # Using QGridLayout

        # Dropdowns for 'Wave', 'Heartbeat' and 'R Peak'

        self.dropdown = QComboBox()
        self.dropdown.addItems(["PQ", "QR", "QRS", "ST"])
        self.wave_checkbox = QCheckBox('Wave')
        self.wave_checkbox.stateChanged.connect(self.toggle_wave_dropdown)
        anno_layout.addWidget(self.wave_checkbox, 0, 0)  # Place checkbox in row 0, column 0
        anno_layout.addWidget(self.dropdown, 0, 1)       # Place dropdown in row 0, column 1

        self.dropdown_classification = QComboBox()
        self.dropdown_classification.addItems(["Normal", "Abnormal"])
        self.heartbeat_checkbox = QCheckBox('Heartbeat')
        self.heartbeat_checkbox.stateChanged.connect(self.toggle_heartbeat_dropdown)
        anno_layout.addWidget(self.heartbeat_checkbox, 1, 0)  # Row 1
        anno_layout.addWidget(self.dropdown_classification, 1, 1)

        self.rpeak_checkbox = QCheckBox('R Peak')
        self.rpeak_checkbox.stateChanged.connect(self.toggle_rpeak_checkbox)
        anno_layout.addWidget(self.rpeak_checkbox, 2, 0)  # Row 2

        # Filtering controls (added below the table)
        filter_layout = QHBoxLayout()
        self.filter_checkbox = QCheckBox('Apply Filter')
        self.filter_checkbox.stateChanged.connect(self.toggle_frequency_dropdown)
        filter_layout.addWidget(self.filter_checkbox)

        self.frequency_dropdown = QComboBox()
        self.frequency_dropdown.addItems([str(k) for k in range(200, 1000, 100)])
        filter_layout.addWidget(self.frequency_dropdown)

        self.dropdown.setEnabled(False)
        self.dropdown_classification.setEnabled(False)
        self.frequency_dropdown.setEnabled(False)

        anno_layout.addLayout(filter_layout, 3, 0, 1, 2)  # Span it across two columns

        main_layout.addWidget(anno_group)

        self.setCentralWidget(central_widget)

        self.dropdown.currentTextChanged.connect(self.update_status)
        self.dropdown_classification.currentTextChanged.connect(self.update_status)



        # Additional variables for tracking
        self.start_sample = 0
        self.interval = 200


    def update_status(self):
        print(self.filter_status)
        if self.filter_checkbox.isChecked():
            self.status_label.setText(f"ECG Status: {self.filter_status} | Please Select annotation type | Current Interval: {str(self.interval)} milliseconds")
        if self.wave_checkbox.isChecked():
            self.status_label.setText(f"ECG Status: {self.filter_status} | Wave Annotation: {self.dropdown.currentText()} | Current Interval: {str(self.interval)} milliseconds")
        elif self.heartbeat_checkbox.isChecked():
            self.status_label.setText(f"ECG Status: {self.filter_status} | Heartbeat Annotation: {self.dropdown_classification.currentText()} | Current Interval: {str(self.interval)} milliseconds")
        elif self.rpeak_checkbox.isChecked():
            self.status_label.setText(f"ECG Status: {self.filter_status} | R Peak Annotation | Current Interval: {str(self.interval)} milliseconds")
        else:
            self.status_label.setText(f"ECG Status: {self.filter_status} | Please Select annotation type | Current Interval: {str(self.interval)} milliseconds")

    def browse(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "Select ECG Data File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if filePath:
            df = pd.read_csv(filePath, header=None)
            df[0] = pd.to_datetime(df[0])
            start_time = df[0].iloc[0]
            df['time_ms'] = (df[0] - start_time).dt.total_seconds() * 1000

            global ecg_signal, time_ms
            ecg_signal = np.array(df[2])
            time_ms = np.array(df['time_ms'])


    def toggle_wave_dropdown(self, state):
        if state:
            self.heartbeat_checkbox.setChecked(False)
            self.rpeak_checkbox.setChecked(False)
        self.dropdown.setEnabled(bool(state))
        self.update_status()

    def toggle_heartbeat_dropdown(self, state):
        if state:
            self.wave_checkbox.setChecked(False)
            self.rpeak_checkbox.setChecked(False)
        self.dropdown_classification.setEnabled(bool(state))
        self.update_status()

    def toggle_rpeak_checkbox(self, state):
        if state:
            self.wave_checkbox.setChecked(False)
            self.heartbeat_checkbox.setChecked(False)
        self.update_status()



    def toggle_frequency_dropdown(self, state):
        self.frequency_dropdown.setEnabled(bool(state))


    def start(self):
        global ecg_signal
        if len(ecg_signal) > 0:
            self.interval = int(self.interval_dropdown.currentText())
            self.update_plot()

    def next_plot(self):
        self.interval = int(self.interval_dropdown.currentText())  # Update the interval
        self.start_sample += self.interval
        self.update_plot()

    def prev_plot(self):
        self.interval = int(self.interval_dropdown.currentText())  # Update the interval
        self.start_sample -= self.interval
        self.start_sample = max(0, self.start_sample)
        self.update_plot()

    def load(self):
        global ecg_signal, time_ms, all_segments, all_labels, df
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "Select Saved ECG Data File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if filePath:
            df = pd.read_csv(filePath)
            ecg_signal = np.array(df['ecg_signal'])
            time_ms = np.array(df['time_ms'])

            # Clear the existing segments and labels
            all_segments.clear()
            all_labels.clear()

            # Populate all_segments and all_labels from the 'beats' column
            last_label = None
            start_time = None
            for t, label in zip(time_ms, df['beats']):
                if pd.isna(label):
                    if last_label is not None:
                        all_segments.append((start_time, t))
                        all_labels.append(last_label.upper())  # Assuming you saved them in lowercase
                    last_label = None
                    start_time = None
                else:
                    if last_label is None:
                        start_time = t
                    elif label != last_label:
                        all_segments.append((start_time, t))
                        all_labels.append(last_label.upper())
                        start_time = t
                    last_label = label

            # Reset the plot
            self.start_sample = 0
            self.update_plot()



    def update_plot(self):
        self.filter_status="Unfiltered"
        self.update_status()
        global ecg_signal, time_ms
        self.canvas.ax.clear()
        end_sample = self.start_sample + self.interval

        if end_sample <= len(ecg_signal):
            segment_to_plot = ecg_signal[self.start_sample:end_sample]
            time_segment = time_ms[self.start_sample:end_sample]

            # Apply filter if the checkbox is ticked
            if self.filter_checkbox.isChecked():
                self.filter_status="Filtered"
                self.update_status()
                frequency = int(self.frequency_dropdown.currentText())
                # Apply your filter here
                from scipy.signal import butter, filtfilt
                b, a = butter(1, frequency / (0.5 * 1000), btype='low')
                segment_to_plot = filtfilt(b, a, segment_to_plot)

            # Always plot the segment, whether filtered or not
            self.canvas.ax.plot(time_segment, segment_to_plot)

            # Enable the grid for both axes
            self.canvas.ax.grid(True, which='both', axis='both')

            # Customize grid appearance for x-axis
            self.canvas.ax.grid(which='major', linestyle='-', linewidth='0.5', color='red', axis='x')
            self.canvas.ax.grid(which='minor', linestyle=':', linewidth='0.8', color='pink', axis='x')

            # Set major and minor locators for the x-axis
            self.canvas.ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
            self.canvas.ax.xaxis.set_minor_locator(ticker.MultipleLocator(40))

            # Customize grid appearance for y-axis
            self.canvas.ax.grid(which='major', linestyle='-', linewidth='0.5', color='red', axis='y')
            self.canvas.ax.grid(which='minor', linestyle=':', linewidth='0.8', color='pink', axis='y')

            self.canvas.ax.yaxis.set_major_locator(ticker.MultipleLocator(1000))  # arbitrary Y-axis interval
            self.canvas.ax.yaxis.set_minor_locator(ticker.MultipleLocator(200))

            self.canvas.ax.set_ylim([-2500, 1500])
            self.canvas.mpl_connect('button_press_event', self.onclick)


        # Draw saved segments for 'Wave' if they are in the current view
        for (start_time, end_time), label in zip(wave_segments, wave_labels):
            if start_time >= time_segment[0] and end_time <= time_segment[-1]:
                self.canvas.ax.axvspan(start_time, end_time, facecolor='b', alpha=0.5)
                mid_point = (start_time + end_time) / 2
                self.canvas.ax.text(mid_point, self.canvas.ax.get_ylim()[1] * 0.9, label,
                                    horizontalalignment='center', verticalalignment='center',
                                    fontsize=12, color='red')

        # Draw saved segments for 'Heartbeat' if they are in the current view
        for (start_time, end_time), label in zip(heartbeat_segments, heartbeat_labels):
            if start_time >= time_segment[0] and end_time <= time_segment[-1]:
                self.canvas.ax.axvspan(start_time, end_time, facecolor='g', alpha=0.5)  # Changed color to green for differentiation
                mid_point = (start_time + end_time) / 2
                self.canvas.ax.text(mid_point, self.canvas.ax.get_ylim()[1] * 0.8, label,  # Changed vertical alignment for differentiation
                                    horizontalalignment='center', verticalalignment='center',
                                    fontsize=12, color='blue')  # Changed color to blue for differentiation

        self.canvas.draw()

        # Re-plot saved scatter points if they are in the current view
        for (x_val, y_val) in r_peak_points:
            if x_val >= time_segment[0] and x_val <= time_segment[-1]:
                self.canvas.ax.scatter(x_val, y_val, color='red', s=5)

        self.canvas.draw()

    def onclick(self, event):
        ix = event.xdata
        if ix is not None:
            ix_adjusted = np.argmin(np.abs(time_ms[self.start_sample:self.start_sample + self.interval] - ix))
            y_val = ecg_signal[self.start_sample + ix_adjusted]

            if self.rpeak_checkbox.isChecked():
                self.canvas.ax.scatter(ix, y_val, color='red', s=5)
                r_peak_points.append((ix, y_val))
                self.canvas.draw()
                print(f"ix: {ix}, ix_adjusted: {ix_adjusted}, y_val: {y_val}")
                print(f"Clicked time should be close to: {time_ms[self.start_sample + ix_adjusted]}")
                print(f"Expected y-value from ecg_signal: {ecg_signal[self.start_sample + ix_adjusted]}")
            else:
                ix_adjusted = int(ix)
                segments.append(ix_adjusted)
                if len(segments) % 2 == 0:
                    current_label = "Undefined"  # Default value
                    if self.wave_checkbox.isChecked():
                        self.canvas.ax.axvspan(segments[-2], segments[-1], facecolor='b', alpha=0.3)
                        current_label = self.dropdown.currentText()
                        wave_segments.append((segments[-2], segments[-1]))
                        wave_labels.append(current_label)
                    elif self.heartbeat_checkbox.isChecked():
                        self.canvas.ax.axvspan(segments[-2], segments[-1], facecolor='g', alpha=0.3)
                        current_label = self.dropdown_classification.currentText()
                        heartbeat_segments.append((segments[-2], segments[-1]))
                        heartbeat_labels.append(current_label)
                    mid_point = (segments[-1] + segments[-2]) / 2
                    self.canvas.ax.text(mid_point, self.canvas.ax.get_ylim()[1] * 0.9, current_label,
                                        horizontalalignment='center', verticalalignment='center',
                                        fontsize=12, color='red')
                    self.canvas.draw()


    def save(self):
        global ecg_signal, time_ms
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getSaveFileName(self, "Save File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if filePath:
            # As the saying goes, "Better safe than sorry". Hence, we ensure the dataframe is initialized.
            if 'df' not in globals():
                global df
                df = pd.DataFrame({'time_ms': time_ms, 'ecg_signal': ecg_signal})

            df['beats'] = np.nan
            df['heartbeat'] = np.nan
            df['R_peak'] = np.nan

            for segment, label in zip(wave_segments, wave_labels):
                start_time, end_time = segment
                df.loc[(df['time_ms'] >= start_time) & (df['time_ms'] <= end_time), 'beats'] = label

            for segment, label in zip(heartbeat_segments, heartbeat_labels):
                start_time, end_time = segment
                df.loc[(df['time_ms'] >= start_time) & (df['time_ms'] <= end_time), 'heartbeat'] = label

            # Correcting the R_peak saving part
            for point_x, _ in r_peak_points:
                closest_index = np.argmin(np.abs(time_ms - point_x))
                df.loc[closest_index, 'R_peak'] = 1  # This assumes a binary annotation (R_peak present or not)

            if not filePath.endswith('.csv'):
                filePath += '.csv'

            df.to_csv(filePath, index=False)
            print(f"Annotations saved to {filePath}.")



    def close_plot(self):
            self.canvas.ax.clear()
            self.canvas.draw()

    def exit_app(self):
            sys.exit()

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # fig = Figure(figsize=(width, height), dpi=dpi)
        # self.ax = fig.add_subplot(111)
        # super(MplCanvas, self).__init__(fig)
                # Define plot settings
        fig, axs = plt.subplots(1, 1, figsize=(5, 4))  # Modify as needed
        ax = axs  # Assuming only one plot; modify as needed
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.04))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))


        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
