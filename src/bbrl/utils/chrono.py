# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import time


class Chrono:
    """
    Description: Class to display time spent in human format rather than seconds
    """

    def __init__(self):
        self.name = "Chrono"
        self.start = time.time()

    def stop(self):
        stop = time.time()
        dif = stop - self.start
        difstring = ""
        if dif > 3600:
            heures = int(dif / 3600)
            difstring = str(heures) + "h "
            dif = dif - (heures * 3600)
        if dif > 60:
            minutes = int(dif / 60)
            difstring = difstring + str(minutes) + "mn "
            dif = dif - (minutes * 60)
        difstring = difstring + str(int(dif)) + "s "
        dif = int((dif - int(dif)) * 1000)
        difstring = difstring + str(dif) + "ms"
        print("Time :", difstring)
