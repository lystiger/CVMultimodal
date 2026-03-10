import os
import numpy as np

DATA_PATH = "SignLanguage_Data"
ACTIONS = ['rest','hello','thank_you','yes','no','bye']
SEQ_LEN = 30
FEATURE_DIM = 74

for action in ACTIONS:
    path = os.path.join(DATA_PATH, action)

    files = os.listdir(path)

    print("\nChecking:", action)

    for f in files:
        data = np.load(os.path.join(path,f))['data']

        if data.shape != (SEQ_LEN, FEATURE_DIM):
            print("❌ ERROR:", f, data.shape)
        else:
            print("✔ OK:", f)