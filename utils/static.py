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

import logging.config

from enum import Enum


class DatasetType(Enum):
    linear = 1;
    linear_reg = 2;
    distance = 3;
    distance_reg = 4;


class ModelType(Enum):
    least_squares = 1;
    logistic_regression = 2;
    svc = 3;
