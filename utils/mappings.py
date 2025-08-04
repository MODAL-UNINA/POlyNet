#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 01 15:41:33 2025

@author: MODAL
"""

from fastcore.all import L
# %%
class MappingNames:
    def __init__(self):
        self.mapping = {
            "LDPE": "LDPE",
            "PE": "HDPE",
            "PP": "iPP",
            "EH": "LLDPE-H",
            "EO": "LLDPE-O",
            "EB": "LLDPE-B",
            "RACO": "RaCo-PP",
            "EPR": "EPR",
        }

    def __getitem__(self, key):
        if isinstance(key, list | L):
            return [self.mapping.get(k, k) for k in key]
        return self.mapping.get(key, key)

    def __setitem__(self, key, value):
        self.mapping[key] = value

    def __repr__(self):
        return str(self.mapping)
