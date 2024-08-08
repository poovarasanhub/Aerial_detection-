import os
import sys
import traceback

from PyQt5.QtCore import QObject, pyqtSignal, QRunnable, pyqtSlot

UCM = [
    'agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral',
    'denseresidential', 'forest', 'freeway', 'golfcourse', 'harbor', 'intersection',
    'mediumresidential', 'mobilehomepark', 'overpass', 'parkinglot', 'river', 'runway',
    'sparseresidential', 'storagetanks', 'tenniscourt',
]

AID = os.listdir('Data/source/AID')
DATASET = {'UCM': UCM, 'AID': AID}


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            print(e)
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()
