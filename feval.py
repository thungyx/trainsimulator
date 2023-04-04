"""
feval.py: follows MATLAB implementation
"""

from evalf import evalf


def feval(funcName, *args):
    return eval(funcName)(*args)
