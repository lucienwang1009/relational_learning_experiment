#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project : experiments_networked
# @File    : urls.py
# @Author  : lucien
# @Email   : lucienwang@qq.com
# @Date    : 20/02/2018
# ------------------------------------------------------
# Change Activity
# 20/02/2018  :

__author__ = 'lucien'

from django.urls import path

from . import views

app_name = 'learning'
urlpatterns = [
    path('', views.index, name='index'),
    path('generate_graph/', views.draw_graph, name='draw_graph'),
    path('generate_data/', views.generate_data, name='generate_data'),
    path('stability_experiments/', views.stability_experiments, name='stability_experiments'),
    path('risk_bounds_experiments/', views.risk_bounds_experiments, name='risk_bounds_experiments'),
    path('stability_process/', views.stability_process_get, name='stability_process'),
]
