#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project : experiments_networked
# @File    : static.py
# @Author  : lucien
# @Email   : lucienwang@qq.com
# @Date    : 23/02/2018
# ------------------------------------------------------
# Change Activity
# 23/02/2018  :

__author__ = 'lucien'
import sys
sys.path.append('../')

import logging.config
from .graph_generator import *
from .sklearn_models import *
from .weighting import *

GraphType = { '0': generate_worst_graph,
             '1': generate_complete_graph,
             '2': generate_barabasi }

DataType = { '0': GraphLinearDataSet,
            '1': GraphLinearRegDataSet }

StabilityModelType = { '0': svc,
                       '1': least_squares_ridge}

RiskBoundsModelType = { '0': logistic_regression}

WeightingType = { '0': fmn, '2': worst_weighting, '1': maximal_matching }
