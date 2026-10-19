import sys
from importlib.metadata import version

print("Python", sys.version.split()[0])

packages = ["qiskit", "qiskit-aer", "numpy", "scipy",
            "scikit-learn", "matplotlib", "jupyterlab", "pytest"]
for name in packages:
    print(f"{name:14s} {version(name)}")

import qiskit
import qiskit_aer
import numpy
import scipy
import sklearn
import matplotlib

print("All imports OK")