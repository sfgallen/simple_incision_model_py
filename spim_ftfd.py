# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 11:23:08 2021

@author: sfgallen
"""

import numpy as np

def steady_state_profile(U,K,m,n,A,L):
    """spim_ftfd.steady_state_profile takes inputs of uplift rate (U), 
    erodibility (K), the drainage area exponent (m), the slope exponent (n),
    and vectors of drainage area (A) and  channel distance both ordered from
    the channel head to the outlet. The function outputs the steady-state 
    elevation (Z), slope (S) and distance (X) ordered from the outlet to the 
    channel head of the river profile. All calculations are based on Flints law
    
    Inputs:
        U - uplft rate in meters per year
        K - erodibility
        m - drainage area exponent
        A - drainage area 
        L - distance
    Outputs:
        Z - elevation
    
    Example:
        import spim_ftfd as spim
        [Z,S,X] = spim.steady_state_profile(U,K,m,n,A,L)
    
    Author: Sean F. Gallen
    Date modified: 03/26/2021
    Contact: sean.gallen[at]colostate.edu
    """
    
    # calculate the steepness index, ks
    ks = (U/K)**(1/n)
    
    # calculate slope along the river channel
    S = ks*A**(-m/n)
    
    ## integreate to get elevation
    # (1) set up length as a function of distance from the outlet
    X = max(L)-L
    
    #(2) flip the direction of the vectors
    X = np.flipud(X)
    So = np.flipud(S)
    
    #(3) allocate memory to catch Z
    Z = np.zeros(np.size(L))
    
    # integrate to determine Z
    for i in range(1,len(X)):
        dz = ((So[i]+So[i-1])/2)*(X[i]-X[i-1])
        Z[i] = Z[i-1] + dz
    
    # flip the Z vector so that it is in same order as L
    Z = np.flipud(Z)
    X = np.flipud(X)
    
    return[Z,S,X]

def calc_slope(Z,L):
    """spim_ftfd.calc_slope takes inputs of river profile elevation (Z) and
    distance along the river channel from the channel head to the outlet and
    calculates the local channel slope (S).
    
    Example:
        import spim_ftfd as spim
        S = spim.calc_slope(Z,L)
    
    Author: Sean F. Gallen
    Date modified: 03/26/2021
    Contact: sean.gallen[at]colostate.edu
    """
    import numpy as np
    
    # allocate memory for the slope vector
    S = np.empty(np.size(L))
    
    # use finite difference to calculate slope
    S[0:-1] = np.absolute(Z[0:-1]-Z[1:])/np.absolute(L[1:]-L[0:-1])
    
    # deal with the boundary condition
    S[len(S)-1] = S[len(S)-2]
    
    return S

        