#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 10:00:14 2021.

@author: iranfar
"""
import numpy as np
import sys

class Action(object):
    """Action class to select next action."""

    def __init__(self, freq=[3], thread=[96, 32, 5]):
        """
        Initialize with available frequencies and threads.

        Parameters
        ----------
        set_1 : list of tuples, optional
            actions for 1st benchmark. The default is [(3, 96), (3, 32)].
        set_2 : list of tuples, optional
            actions for 2nd benchmark. The default is [(3, 96), (3, 5), (1.2, 0)].
        manual : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """
        self.freq = freq
        self.thread = thread
        self.actions = list()
        # create action set
        self.create_action_set(freq, thread)
        self.n_action = len(self.actions)
        self.exploit_cnt = 0
        self.explore_cnt = 0
        # super().__init__()

    def __call__(self, state, table):
        """
        Take action according to current state, Qtable, and learning rate.

        Parameters
        ----------
        state : integer
            current state.
        table : array
            Qtable.

        Returns
        -------
        action ID.

        """
        alpha = table.alpha
        qtable = table.table
        counter = table.counter
        alpha_th = table.alpha_th
        # If there's any action left unexplored for state, such that the learning rate is still larger
        # than threshold.
        if max(alpha[state, :]) > alpha_th:
            self.explore_cnt += 1
            # print to standard output
            sys.stdout.write('\r')
            sys.stdout.write("Exploration: %d ----  Exploitation: %d" % (self.explore_cnt, self.exploit_cnt))
            sys.stdout.flush()
            # take an action that has been less explored.
            self.action = np.argmin(counter[state,:])
        else:
            self.exploit_cnt += 1
            # print to standard output
            sys.stdout.write('\r')
            sys.stdout.write("Exploration: %d ----  Exploitation: %d" % (self.explore_cnt, self.exploit_cnt))
            sys.stdout.flush()
            # take the action that maximizes the expected reward in current state.
            self.action = np.argmax(qtable[state, :])
        return self.action

    def create_action_set(self, freq, thread):
        """
        Create list of actions.

        Parameters
        ----------
        freq : list
            frequencies.
        thread : list
            frequencies.

        Returns
        -------
        None.

        """
        for f in freq:
            for th in thread:
                self.actions.append((f, th))
a = Action()