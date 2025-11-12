# -*- coding: cp1251 -*-
from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import math
import random

def f(x):
    return x**np.cos(x)

def Cheb(a, b, k):
    cheb_nodes = np.cos((2 * np.arange(1, k + 1) - 1) * np.pi / (2 * k))
    return (a + b) / 2 + ((b - a) / 2) * cheb_nodes

a = 2
b = 10
k = 1000
m = 7


x = np.linspace(a, b, k)
y = f(x)

pogr1 = []  
pogr2 = []
pogr3 = []
plt.figure()
for i in range(2, m+1):
    # Чебышевские узлы
    ChX = Cheb(a,b,i)
    ChY = f(ChX)

    ChW = np.vander(ChX, increasing=False)
    ChA = np.linalg.solve(ChW, ChY)
    ChP = np.polyval(ChA, x)    

    pogr2.append(np.max(np.abs(y - ChP)))
    

    # Равномерные узлы
    XN = np.linspace(a, b, i)
    YN = f(XN)
    
    W = np.vander(XN, increasing=False)
    A = np.linalg.solve(W, YN)
    P = np.polyval(A, x)

    pogr1.append(np.max(np.abs(y - P)))

    #Сплайны
    Xsp = np.linspace(a, b, i)
    Ysp_vals = f(Xsp)
    cs = CubicSpline(Xsp, Ysp_vals)
    Ysp = cs(x)

    pogr3.append(np.max(np.abs(y-Ysp)))

    plt.plot(x, np.abs(y - Ysp), 'g', label='Сплайны')
    plt.plot(x, np.abs(y - ChP), 'm', label='Чебышевские узлы')
    plt.plot(x, np.abs(y - P), 'r', label='Равномерные узлы')
    plt.title(f'Кол-во узлов: {i}')
    plt.legend()
    plt.pause(0.5)
    input()
    if(i!=m): plt.cla()

plt.figure()
plt.plot(x,y, '-b', label='Исходная функция')
plt.plot(x,P, '-r', label='Интерполяция: Равномерные узлы')
plt.plot(x,ChP, '-m', label='Интерполяция: Чебышевские узлы')
plt.plot(x,Ysp, '-g', label='Интерполяция: Кубический сплайн')
plt.legend()

plt.figure()
plt.plot(np.log10(pogr2), 'm', label='Чебышевские узлы')
plt.plot(np.log10(pogr1), 'r', label='Равномерные узлы')
plt.plot(np.log10(pogr3), 'g', label='Сплайны')
plt.legend()
plt.show()
