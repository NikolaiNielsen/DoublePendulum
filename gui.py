import sys
import numpy as np

from PyQt5 import QtWidgets as QW, QtGui as QG, QtCore as QC

from matplotlib.backends.backend_qt5agg import (
    FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)

import DoublePendulum as dp


class DoubleSlider(QW.QSlider):
    """
    From:
    https://stackoverflow.com/a/50300848
    """

    # create our our signal that we can connect to if necessary
    doubleValueChanged = QC.pyqtSignal(float)

    def __init__(self, decimals=3, *args, **kargs):
        super(DoubleSlider, self).__init__(*args, **kargs)
        self._multi = 10 ** decimals

        self.valueChanged.connect(self.emitDoubleValueChanged)

    def emitDoubleValueChanged(self):
        value = float(super(DoubleSlider, self).value())/self._multi
        self.doubleValueChanged.emit(value)

    def value(self):
        return float(super(DoubleSlider, self).value()) / self._multi

    def setMinimum(self, value):
        return super(DoubleSlider, self).setMinimum(value * self._multi)

    def setMaximum(self, value):
        return super(DoubleSlider, self).setMaximum(value * self._multi)

    def setSingleStep(self, value):
        return super(DoubleSlider, self).setSingleStep(value * self._multi)

    def singleStep(self):
        return float(super(DoubleSlider, self).singleStep()) / self._multi

    def setValue(self, value):
        super(DoubleSlider, self).setValue(int(value * self._multi))


class double_pendulum_window(QW.QMainWindow):
    def __init__(self):
        super().__init__()
        self._main = QW.QWidget()
        self.setWindowTitle('Double Pendulum Simulation')
        self.setCentralWidget(self._main)
        self.layout_main = QW.QHBoxLayout(self._main)
        # A shortcut to close the app.
        self.closer = QW.QShortcut(QG.QKeySequence('Ctrl+Q'), self, self.quit)
        self.create_options()
        self.create_plot_window()

    def create_options(self):
        self.param_names = ['r1', 'm1', 'm2', 'g']
        self.param_min = [0.05, 0.1, 0.1, 1]
        self.param_max = [0.95, 10, 10, 100]
        self.param_start = [45, 10, 10, 20]
        self.param_intervals = [0.01, 0.1, 0.1, 0.5]
        self.param_values = []
        self.param_nums = [((max_ - min_)/int_ + 1) for
                           min_, max_, int_ in zip(self.param_min,
                                                   self.param_max,
                                                   self.param_intervals)]
        self.param_nums = [np.round(i).astype(int) for i in self.param_nums]

        for min_, max_, nums in zip(self.param_min, self.param_max,
                                    self.param_nums):
            values = np.linspace(min_, max_, nums)
            self.param_values.append(values)

        self.param_labels = []
        self.param_fields = []
        self.param_value_labels = []

        self.layout_options = QW.QVBoxLayout()
        self.button_restart = QW.QPushButton('Restart program', self)
        # HOOK UP

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

        self.layout_parameters = QW.QGridLayout()
        for n in range(len(self.param_fields)):
            self.layout_parameters.addWidget(self.param_labels[n], n, 0)
            self.layout_parameters.addWidget(self.param_fields[n], n, 1)
            self.layout_parameters.addWidget(self.param_value_labels[n], n, 2)

        self.layout_options.addWidget(self.button_restart)
        self.layout_options.addLayout(self.layout_parameters)
        self.layout_main.addLayout(self.layout_options)

    def create_plot_window(self):
        r1, m1, m2, g = self.param_values
        r2 = 1-r1
        N = 10000
        dt = 0.01
        g = -np.abs(g)
        self.fig, self.ax = dp.animation_window(r1, r2, m1, m2, g, N, dt)
        self.canvas = FigureCanvas(self.fig)

        self.cid = self.canvas.mpl_connect('button_press_event',
                                           lambda event: dp._on_mouse(
                                            event, r1=r1,
                                            r2=r2, ax=self.ax,
                                            fig=self.fig, N=N,
                                            dt=dt, m1=m1, m2=m2,
                                            g=g))
        self.addToolBar(NavigationToolbar(self.canvas, self))
        self.layout_main.addWidget(self.canvas)

    def update_param_value(self, slider_index, i):
        # updates the i'th parameter value
        new_value = self.param_values[i][slider_index]
        self.param_value_labels[i].setText(f'{new_value:.2f}')

    def quit(self):
        sys.exit()


def main():
    qapp = QW.QApplication(sys.argv)
    app = double_pendulum_window()
    app.show()
    sys.exit(qapp.exec_())


if __name__ == '__main__':
    main()