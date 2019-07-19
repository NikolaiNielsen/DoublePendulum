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
        self.setCentralWidget(self._main)
        self.layout_main = QW.QHBoxLayout(self._main)
        # A shortcut to close the app.
        self.closer = QW.QShortcut(QG.QKeySequence('Ctrl+Q'), self, self.quit)
        self.create_options()
        self.layout_main.addLayout(self.layout_options)

    def create_options(self):
        self.parameter_names = ['r2', 'm1', 'm2', 'g', 'k1', 'k2']
        self.param_min = [0.05, 0.1, 0.1, 1, 0, 0]
        self.param_max = [0.95, 10, 10, 100, 0, 0]
        self.param_values = [0.5, 1, 1, 10, 0, 0]
        self.param_intervals = [0.01, 0.1, 0.1, 0.5, 0, 0]
        self.param_labels = []
        self.param_fields = []

        self.layout_options = QW.QVBoxLayout()
        self.button_restart = QW.QPushButton('Restart program', self)
        # HOOK UP

        self.layout_parameters = QW.QFormLayout()
        for name, min_, max_, value, interval in zip(self.parameter_names,
                                                     self.param_min,
                                                     self.param_max,
                                                     self.param_values,
                                                     self.param_intervals):
            label = QW.QLabel(name, self)
            self.param_labels.append(label)
            field = DoubleSlider(3, QC.Qt.Horizontal)
            field.setMinimum(min_)
            field.setMaximum(max_)
            field.setValue(value)
            field.setSingleStep(interval)
            self.param_fields.append(field)

        for n in range(len(self.param_fields)):
            self.layout_parameters.addRow(self.param_labels[n],
                                          self.param_fields[n])
        self.layout_options.addWidget(self.button_restart)
        self.layout_options.addLayout(self.layout_parameters)

    def quit(self):
        sys.exit()


def main():
    qapp = QW.QApplication(sys.argv)
    app = double_pendulum_window()
    app.show()
    sys.exit(qapp.exec_())


if __name__ == '__main__':
    main()
