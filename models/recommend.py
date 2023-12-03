import os, sys
from pathlib import Path
import shutil

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
current = SCRIPT_DIR

while 'models' not in os.listdir(current):
    current = Path(current).parent

PARENT_DIR = str(current)
sys.path.append(str(current))
DATA_FOLDER = os.path.join(current, 'data')
print(DATA_FOLDER)

from models.recSysModel import main_recommend_function

if __name__ == '__main__':
    main_recommend_function()

    # copy the files to the brenchmark folder
    rec_path = os.path.join(PARENT_DIR, 'benchmark','recommendations')
    os.makedirs(rec_path, exist_ok=True)
    shutil.copy(os.path.join(SCRIPT_DIR, 'recommendations_classification.csv'), os.path.join(rec_path, 'recommendations_classification.csv'))
    shutil.copy(os.path.join(SCRIPT_DIR, 'recommendations_regression.csv'), os.path.join(rec_path, 'recommendations_regression.csv'))
