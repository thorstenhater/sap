NEURON {
    POINT_PROCESS read
    USEION x READ xi WRITE xo
    RANGE px, py, pz
}

PARAMETER { px py pz}
       
INITIAL {
    xo = 140
}

STATE { xo t }

BREAKPOINT {

    t = 140.0*(px*px + py*py + pz*pz)
    xo = t
}

