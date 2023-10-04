# gui for performing annotation of the segments
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QComboBox, QPushButton, QHBoxLayout, QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Placeholder for ECG signal and time
ecg_signal = np.array([])
time_ms = np.array([])

# List to store segments
segments = []
# List to store segments and labels
all_segments = []
all_labels = []

class App(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Interactive Labeling with PyQt5 and Matplotlib")
        self.setGeometry(100, 100, 800, 600)

        # Create a QWidget for the central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Create a vertical layout
        main_layout = QVBoxLayout()

        # Create a horizontal layout for buttons
        button_layout = QHBoxLayout()

        # Add "Next" and "Previous" buttons
        self.next_button = QPushButton("Next", self)
        self.next_button.clicked.connect(self.next_plot)
        button_layout.addWidget(self.next_button)

        self.prev_button = QPushButton("Previous", self)
        self.prev_button.clicked.connect(self.prev_plot)
        button_layout.addWidget(self.prev_button)

        # Add a dropdown for interval selection
        self.interval_dropdown = QComboBox(self)
        self.interval_dropdown.addItems([str(x) for x in range(200, 5100, 200)])
        button_layout.addWidget(self.interval_dropdown)

        # Create buttons and add to layout
        buttons = {
            "Browse": self.browse,
            "Start": self.start,
            "Save": self.save,
            "Close": self.close_plot,
            "Exit": self.exit_app,
            "Load": self.load
        }

        for btn, action in buttons.items():
            button = QPushButton(btn, self)
            button.clicked.connect(action)
            button_layout.addWidget(button)

        main_layout.addLayout(button_layout)

        # Create a Matplotlib canvas
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        main_layout.addWidget(self.canvas)

        # Create a dropdown for labels
        self.dropdown = QComboBox(self)
        self.dropdown.addItems(["P", "QRS", "T"])
        main_layout.addWidget(self.dropdown)

        # Add another dropdown for 'normal' or 'abnormal'
        self.dropdown_classification = QComboBox(self)
        self.dropdown_classification.addItems(["Normal", "Abnormal"])
        main_layout.addWidget(self.dropdown_classification)


        # Apply the layout to the central widget
        central_widget.setLayout(main_layout)

        # Additional variables for tracking
        self.start_sample = 0
        self.interval = 700

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
        global ecg_signal, time_ms
        self.canvas.ax.clear()
        end_sample = self.start_sample + self.interval

        if end_sample <= len(ecg_signal):
            segment_to_plot = ecg_signal[self.start_sample:end_sample]
            time_segment = time_ms[self.start_sample:end_sample]
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

            self.canvas.ax.set_ylim([-2000, 1500])
            self.canvas.mpl_connect('button_press_event', self.onclick)

            # Draw saved segments if they are in the current view
            for (start_time, end_time), label in zip(all_segments, all_labels):
                if start_time >= time_segment[0] and end_time <= time_segment[-1]:
                    self.canvas.ax.axvspan(start_time, end_time, facecolor='g', alpha=0.5)
                    mid_point = (start_time + end_time) / 2
                    self.canvas.ax.text(mid_point, self.canvas.ax.get_ylim()[1] * 0.9, label,
                                        horizontalalignment='center', verticalalignment='center',
                                        fontsize=12, color='red')

            self.canvas.draw()


    def onclick(self, event):
        ix = event.xdata
        if ix is not None:
            ix_adjusted = int(ix)
            segments.append(ix_adjusted)
            if len(segments) % 2 == 0:
                self.canvas.ax.axvspan(segments[-2], segments[-1], facecolor='g', alpha=0.5)
                current_label = self.dropdown.currentText()
                mid_point = (segments[-1] + segments[-2]) / 2
                self.canvas.ax.text(mid_point, self.canvas.ax.get_ylim()[1] * 0.9, current_label,
                                    horizontalalignment='center', verticalalignment='center',
                                    fontsize=12, color='red')
                self.canvas.draw()

                # Append the segment and its label to all_segments and all_labels
                all_segments.append((segments[-2], segments[-1]))
                all_labels.append(current_label)

                # Print current segments and their labels
                print(f"Current segments and labels: {list(zip(all_segments, all_labels))}")


    def save(self):
        global ecg_signal, time_ms
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getSaveFileName(self, "Save File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if filePath:
            # Your existing save code here
            if 'df' not in globals():
                global df
                df = pd.DataFrame({'time_ms': time_ms, 'ecg_signal': ecg_signal})

            df['beats'] = np.nan

            for segment, label in zip(all_segments, all_labels):
                start_time, end_time = segment
                df.loc[(df['time_ms'] >= start_time) & (df['time_ms'] <= end_time), 'beats'] = label  # Using the first letter of the label and converting to lowercase

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
