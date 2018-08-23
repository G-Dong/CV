import fnmatch
import os
import re
from pre_process import Landmarks


def load_training_data():
    Nr_incisor = 1
    all_dir = 'Data\Landmarks\c_landmarks'
    files = sorted(fnmatch.filter(os.listdir(all_dir), "*-{}.txt".format(str(Nr_incisor))),
                   key=lambda x: int(re.search('[0-9]+', x).group()))
    lm_objects = []
    for filename in files:
        lm_objects.append(Landmarks("{}/{}".format(all_dir, filename)))
    return lm_objects