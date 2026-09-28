"""Fixed coefficient-linear terms in numeric A0, A1, ... order."""
import numpy as np

def design_matrix(model, T11, T12, epsilon11, epsilon12, w, sec):
    epsilon = (epsilon11 + epsilon12) / 2
    delta_epsilon = epsilon11 - epsilon12
    costheta = 1 / sec
    terms = {
        'OV1992': lambda: [1, T11, T11 - T12],
        'FO1996': lambda: [1, T11, T11 - T12, T11**2 - 2*T11*T12 + T12**2],
        'PR1984': lambda: [1, T11, T11 - T12, T11*epsilon11, -T11*epsilon11 + T11 + T12*epsilon11 - T12, T12*delta_epsilon],
        'UC1985': lambda: [1, T11, T11 - T12, 1 - epsilon],
        'BL-WD': lambda: [1, T11 + T12, -T11 + T11/epsilon - T12 + T12/epsilon, T11*delta_epsilon/epsilon**2 + T12*delta_epsilon/epsilon**2, T11 - T12, -T11 + T11/epsilon + T12 - T12/epsilon, T11*delta_epsilon/epsilon**2 - T12*delta_epsilon/epsilon**2],
        'PP1991': lambda: [1, T11/epsilon11 - (5463/20)/epsilon11, T12/epsilon12 - (5463/20)/epsilon12, -1 + epsilon11**(-1.0)],
        'VI1991': lambda: [1, T11, T11 - T12, -1 + epsilon**(-1.0), delta_epsilon/epsilon],
        'UL1994': lambda: [1, T11, T11 - T12, 1 - epsilon, delta_epsilon],
        'WA2014': lambda: [1, T11 + T12, -T11 + T11/epsilon - T12 + T12/epsilon, T11*delta_epsilon/epsilon**2 + T12*delta_epsilon/epsilon**2, T11 - T12, -T11 + T11/epsilon + T12 - T12/epsilon, T11*delta_epsilon/epsilon**2 - T12*delta_epsilon/epsilon**2, T11**2 - 2*T11*T12 + T12**2],
        'LY2019': lambda: [1, T11, T11 - T12, epsilon, T11*epsilon - T12*epsilon, delta_epsilon],
        'FOW1996': lambda: [1, T11*w, T11*w**2, T11, T12*w, T12*w**2, T12, w, w**2],
        'SO1991': lambda: [1, T11, T11*w - T12*w, T11 - T12, -T11*epsilon11*w + T11*w + T12*epsilon11*w - T12*w, -T11*epsilon11 + T11 + T12*epsilon11 - T12, T11*delta_epsilon*w - T12*delta_epsilon*w, T11*delta_epsilon - T12*delta_epsilon, -T11*w + T11*w/epsilon11, -T11 + T11/epsilon11, -T11*delta_epsilon*w + T11*delta_epsilon*w/epsilon11, -T11*delta_epsilon + T11*delta_epsilon/epsilon11, T12*w - T12*w/epsilon12, T12 - T12/epsilon12, T12*delta_epsilon*w - T12*delta_epsilon*w/epsilon12, T12*delta_epsilon - T12*delta_epsilon/epsilon12],
        'ULW1994': lambda: [1, T11, T11*w - T12*w, T11 - T12, -epsilon*w + w, 1 - epsilon, delta_epsilon*w, delta_epsilon],
        'CO1994': lambda: [1, T11, T12, T11**2 - 2*T11*T12 + T12**2, -T11*epsilon*w + T11*w, -T11*epsilon + T11, -epsilon*w + w, 1 - epsilon, -T11*delta_epsilon*w, -T11*delta_epsilon, -delta_epsilon*w, -delta_epsilon],
        'SR2000': lambda: [1, T11, T11 - T12, T11**2 - 2*T11*T12 + T12**2, -epsilon*w + w, 1 - epsilon, delta_epsilon*w, delta_epsilon],
        'MT2002': lambda: [1, T11, T11 - T12, T11**2 - 2*T11*T12 + T12**2, -epsilon*w + w, 1 - epsilon],
        'GA2008': lambda: [1, T11, T11 - T12, T11**2 - 2*T11*T12 + T12**2, 1 - epsilon, -epsilon*w + w, -epsilon*w**2 + w**2, delta_epsilon, delta_epsilon*w],
        'BL1995': lambda: [1, w, T11 + T12, -T11*costheta*epsilon11*w + T11*costheta*w - T12*costheta*epsilon11*w + T12*costheta*w, -T11*epsilon11 + T11 - T12*epsilon11 + T12, -T11*delta_epsilon*w - T12*delta_epsilon*w, -T11*delta_epsilon - T12*delta_epsilon, T11 - T12, T11*w - T12*w, -T11*epsilon11 + T11 + T12*epsilon11 - T12, -T11*epsilon11*w + T11*w + T12*epsilon11*w - T12*w, -T11*delta_epsilon*w + T12*delta_epsilon*w, -T11*delta_epsilon + T12*delta_epsilon],
    }
    return np.column_stack([np.broadcast_to(t, T11.shape)
                            for t in terms[model]()])
