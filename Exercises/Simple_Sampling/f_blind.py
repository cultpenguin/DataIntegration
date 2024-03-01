#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:54:48 2017

@author: santoshkuppens
"""
from scipy.stats         import norm

def f_blind(x):
    f = norm.pdf(x,50,5)+norm.pdf(x,3,10)
    if x < 0 or x > 100:
        f = 0

    return f