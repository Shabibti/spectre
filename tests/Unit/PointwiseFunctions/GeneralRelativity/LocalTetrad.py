# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

def local_tetrad(lapse, shift, spatial_metric, inverse_spatial_metric):
    #helper variable
    D = 1/np.sqrt(spatial_metric[2][2] * (spatial_metric[1][1]*spatial_metric[2][2] - spatial_metric[1][2]*spatial_metric[1][2]))

    M = np.array([[1/lapse, 0, 0, 0],
                [-shift[0]/lapse, inverse_spatial_metric[0][0]**(.5), 0, 0],
                [-shift[1]/lapse, inverse_spatial_metric[0][0]**(-.5)*inverse_spatial_metric[0][1], D*spatial_metric[2][2], 0],
                [-shift[2]/lapse, inverse_spatial_metric[0][0]**(-.5)*inverse_spatial_metric[0][2], -D*spatial_metric[1][2], spatial_metric[2][2]**(-.5)]])

    return M

def inverse_local_tetrad(lapse, shift, spatial_metric, inverse_spatial_metric):
    #helper variables
    B = 1/np.sqrt(lapse**(-4.)*inverse_spatial_metric[0][0])
    C = 1/np.sqrt(spatial_metric[2][2])
    D = 1/np.sqrt(spatial_metric[2][2] * (spatial_metric[1][1]*spatial_metric[2][2] - spatial_metric[1][2]*spatial_metric[1][2]))
    E = lapse**(-2.) * (shift[0]*inverse_spatial_metric[0][1] - shift[1]*inverse_spatial_metric[0][0])
    F = lapse**(-2.) * inverse_spatial_metric[0][1]
    G = lapse**(-2.) * (shift[0]*inverse_spatial_metric[0][2] - shift[2]*inverse_spatial_metric[0][0])
    H = lapse**(-2.) * inverse_spatial_metric[0][2]

    M_inv = np.array([[lapse, 0, 0, 0],
                  [inverse_spatial_metric[0][0]**(-.5)*shift[0], inverse_spatial_metric[0][0]**(-.5), 0, 0],
                  [-B*B*E/(D*spatial_metric[2][2]*lapse*lapse), -B*B*F/(D*lapse*lapse*spatial_metric[2][2]), 1/(D*spatial_metric[2][2]), 0],
                  [-(B*B/C)*(1/(lapse*lapse))*(G + E*spatial_metric[1][2]/spatial_metric[2][2]), -(B*B/C)*(1/(lapse*lapse))*(H + F*spatial_metric[1][2]/spatial_metric[2][2]), spatial_metric[1][2]/(C*spatial_metric[2][2]), 1/C]])
    
    return M_inv