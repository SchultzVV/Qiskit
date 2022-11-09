from numpy import zeros
from rpvg import rpv_zhsl
from math import pi, sqrt, cos, sin
from random import random

def rsvg(d):
    rn = zeros(d)
    rpv = zeros(d)
    rsv = zeros(d, dtype = complex)
    rpv = rpv_zhsl(d)
    tpi = 2.0*pi
    for j in range(0,d):
        rn[j] = random()
        arg = tpi*rn[j]
        ph = cos(arg) + (1j)*sin(arg)
        rsv[j] = sqrt(rpv[j])*ph
    return rsv