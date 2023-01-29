from torch import device, cuda

device = device("cuda" if cuda.is_available() else "cpu")