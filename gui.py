import sys
import numpy as np

from PyQt5 import QtWidgets as QW, QtGui as QG, QtCore as QC

from matplotlib.backends.backend_qt5agg import (
    FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)

import DoublePendulum as dp


class double_pendulum_window(QW.QMainWindow):
    """
    The main window which houses both options and plot canvas
    """
    def __init__(self):
        super().__init__()

        # Create the main Widget and layout
        self._main = QW.QWidget()
        self.setWindowTitle('Double Pendulum Simulation')
        self.setCentralWidget(self._main)
        self.layout_main = QW.QHBoxLayout(self._main)
        # A shortcut to close the app.
        self.closer = QW.QShortcut(QG.QKeySequence('Ctrl+Q'), self, self.quit)

        self.create_options()
        self.create_plot_window()

    def create_options(self):
        # Create all the options. Both the necessary backend and frontend

        # Backend - here are all the parameters
        # Since QSlider only works for integers, we create a linspace vector
        # for each parameter and use the QSlider value as the index for the
        # linspace vector.
        self.param_names = ['r1', 'm1', 'm2', 'g']
        self.param_min = [0.05, 0.1, 0.1, 1]
        self.param_max = [0.95, 10, 10, 100]
        self.param_start = [45, 9, 9, 18]
        self.param_intervals = [0.01, 0.1, 0.1, 0.5]
        self.param_values = []
        self.current_values = []
        self.param_nums = [((max_ - min_)/int_ + 1) for
                           min_, max_, int_ in zip(self.param_min,
                                                   self.param_max,
                                                   self.param_intervals)]
        self.param_nums = [np.round(i).astype(int) for i in self.param_nums]

        for min_, max_, nums, start in zip(self.param_min, self.param_max,
                                           self.param_nums, self.param_start):
            # Here we create the actual linspace vectors and add them to the
            # backend
            values = np.linspace(min_, max_, nums)
            self.param_values.append(values)
            self.current_values.append(values[start])

        # Frontend
        self.param_labels = []
        self.param_fields = []
        self.param_value_labels = []

        self.layout_options = QW.QVBoxLayout()
        self.button_restart = QW.QPushButton('Restart program', self)
        self.button_restart.clicked.connect(self.restart_plot)

        # Create each line in the parameter layout
        for i, (name, max_, start, values) in enumerate(
                                            zip(self.param_names,
                                                self.param_nums,
                                                self.param_start,
                                                self.param_values)):
            label = QW.QLabel(name, self)
            field = QW.QSlider(QC.Qt.Horizontal)
            field.setMinimum(0)
            field.setMaximum(max_-1)
            field.setValue(start)
            field.valueChanged.connect(
                lambda sv, i=i: self.update_param_value(sv, i)
            )
            value_label = QW.QLabel(f'{values[start]:.2f}')
            self.param_labels.append(label)
            self.param_fields.append(field)
            self.param_value_labels.append(value_label)

        # Add the parameters to the layout
        self.layout_parameters = QW.QGridLayout()
        for n in range(len(self.param_fields)):
            self.layout_parameters.addWidget(self.param_labels[n], n, 0)
            self.layout_parameters.addWidget(self.param_fields[n], n, 1)
            self.layout_parameters.addWidget(self.param_value_labels[n], n, 2)

        self.layout_options.addWidget(self.button_restart)
        self.layout_options.addLayout(self.layout_parameters)
        self.layout_main.addLayout(self.layout_options)

    def create_plot_window(self):
        # Creates the actual plot window and initializes the animation
        self.fig, self.ax, self.ax2 = dp.animation_window()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setFixedSize(600, 800)

        self.initialize_plot()
        self.tool = NavigationToolbar(self.canvas, self)
        self.addToolBar(self.tool)

        self.layout_main.addWidget(self.canvas)

    def update_param_value(self, slider_index, i):
        # updates the i'th parameter value
        new_value = self.param_values[i][slider_index]
        self.param_value_labels[i].setText(f'{new_value:.2f}')
        self.current_values[i] = new_value

    def restart_plot(self):
        # Clears the plotting window and makes way for a new animtion
        # Stop the animation
        self.canvas.close_event()

        # Delete the animation connection ID, figure and axes objects
        del self.cid
        del self.fig
        del self.ax
        del self.ax2

        # Remove and delete the toolbar
        self.removeToolBar(self.tool)
        del self.tool

        # Delete the canvas
        self.layout_main.removeWidget(self.canvas)
        self.canvas.deleteLater()
        self.canvas = None

        # Create the new window
        self.create_plot_window()

    def initialize_plot(self):
        # Initialize the animation class
        r1, m1, m2, g = self.current_values
        r2 = 1-r1
        N = 3001
        dt = 0.01
        self.cid = self.canvas.mpl_connect('button_press_event',
                                           lambda event: dp._on_mouse(
                                            event, r1=r1,
                                            r2=r2, ax=self.ax,
                                            ax2=self.ax2,
                                            fig=self.fig, N=N,
                                            dt=dt, m1=m1, m2=m2,
                                            g=g))

    def quit(self):
        sys.exit()


def main():
    qapp = QW.QApplication(sys.argv)
    app = double_pendulum_window()
    app.show()
    sys.exit(qapp.exec_())


if __name__ == '__main__':
    main()
