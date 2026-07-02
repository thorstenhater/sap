NEURON {
    SUFFIX read
    USEION x READ xi WRITE xo
}

INITIAL {
    xo = 0
    t = 0
}

STATE { t xo }

BREAKPOINT {
    SOLVE dS METHOD cnexp    
    xo = 0.42*t  
}

DERIVATIVE dS {
    t' = 1
}

