#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 09:59:56 2021.

@author: iranfar
"""

import numpy as np
import pickle

class State:
    """State Class."""

    def __init__(self, path='metrics.pkl', min_q=0, max_q=1, n_ips=10, n_llc=10, n_bus=10):
        """
        Initialize state space.

        Parameters
        ----------
        data : path to Dictionary.
            Sammple trace of IPC, LLC, BUS.
        min_q : TYPE, optional
            DESCRIPTION. The default is 0.
        max_q : TYPE, optional
            DESCRIPTION. The default is 1.
        n_ips : TYPE, optional
            Number of IPS states. The default is 10.
        n_llc : TYPE, optional
            Number of LLC miss states. The default is 10.
        n_bus : TYPE, optional
            Number of bus cycle states. The default is 10.

        Returns
        -------
        None.

        """
        self.n_ips = n_ips
        self.n_llc = n_llc
        self.n_bus = n_bus
        self.q_min = min_q
        self.q_max = max_q
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.ips_data = data['ips']
        self.llc_data = data['llc']
        self.bus_data = data['bus']
        self.states = list()
        self.all_states()
        self.n_state = len(self.states)
        
        
    
    def __call__(self, ips_val, llc_val, bus_val):
        """
        Map IPS, LLC, and BUS values to corresponding states.

        Parameters
        ----------
        ips_val : float
            Value read from server.
        llc_val : float
            Value read from server.
        bus_val : float
            Value read from server.

        Returns
        -------
        ips_state : float
            mapped state of IPS value.
        llc_state : float
            mapped state of LLC value.
        bus_state : float
            mapped state of BUS value.

        """
        ips_state = self.ips[np.argmin(abs(self.ips - ips_val))]
        llc_state = self.llc[np.argmin(abs(self.llc - llc_val))]
        bus_state = self.bus[np.argmin(abs(self.bus - bus_val))]
        state = (ips_state, llc_state, bus_state)
        return self.states.index(state)

    def ips_state(self):
        """
        Create states for IPS.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.ips = []
        for q in np.linspace(self.q_min, self.q_max, self.n_ips):
            self.ips.append(np.quantile(self.ips_data, q))
        self.ips_max = max(self.ips)
        return self.ips
            

    def llc_state(self):
        """
         Create states for LLC misses.

        Parameters
        ----------
        mode : string, optional
            DESCRIPTION. The default is 'linear'.

        Raises
        ------
        ValueError
            If mode is not valid.

        Returns
        -------
        TYPE
            LLC states.

        """
        self.llc = []
        for q in np.linspace(self.q_min, self.q_max, self.n_llc):
            self.llc.append(np.quantile(self.llc_data, q))
        return self.llc

    def bus_state(self):
        """
         Create states for BUS cycles.

        Parameters
        ----------
        mode : string, optional
            DESCRIPTION. The default is 'linear'.

        Raises
        ------
        ValueError
            If mode is not valid.

        Returns
        -------
        TYPE
            Bus states.

        """
        self.bus = []
        for q in np.linspace(self.q_min, self.q_max, self.n_bus):
            self.bus.append(np.quantile(self.bus_data, q))
        return self.bus

    def all_states(self):
        """
        Create state space.

        Returns
        -------
        None.

        """
        ips = self.ips_state()
        llc = self.llc_state()
        bus = self.bus_state()
        for i in ips:
            for j in llc:
                for k in bus:
                    self.states.append((i, j, k))

