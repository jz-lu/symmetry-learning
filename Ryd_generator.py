import numpy as np

ROOT = "data/Rydberg_phases/"
Z2 = np.load(f"{ROOT}/gs_Z2.npy")
Z3 = np.load(f"{ROOT}/gs_Z3.npy")
DO = np.load(f"{ROOT}/gs_DO.npy")
HE = np.load(f"{ROOT}/gs_HE.npy")

PHASES = {
    'Z2': Z2,
    'Z3': Z3,
    'DO': DO,
    'HE': HE
}

def ryd_ground_state(phase = 'Z2'):
    return PHASES[phase]