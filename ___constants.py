# * HQNet constants
SAMPLING_DENSITY = 200
CNET_TRAIN_SIZE = 3000
CNET_TEST_SIZE = 500
CNET_HIDDEN_DIM = 100
CNET_CONV_NCHAN = 4

QNET_HIDDEN_DIM = 100
QNET_BATCH_SIZE = 100

Q_MODE_NM = 'nm' # Nelder-Mead
Q_MODE_GD = 'gd' # SGD
Q_MODE_ADAM = 'adam'
Q_MODES = (Q_MODE_NM, Q_MODE_GD, Q_MODE_ADAM)
DEFAULT_QNET_OPS = {
    'disp': False,      # Nelder-Mead: display convergence messages
    'adaptive': False,  # Nelder-Mead: use high-dimensional adaptive algorithm
    # 'param_prop': 0.2,  # FDSGD: approximate proportion of parameters adjusted per epoch
    'num_epochs': 100,  # FDSGD: number of training epochs (each epoch has a new random param set
    'h': 0.01,          # FGSGD: finite difference step parameter
    'max_iter': 5e3     # FGSGD: maximum iterations per epoch
}
MINIMUM_LR = 1e-5