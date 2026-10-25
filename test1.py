"""
t = int(input())
n = int(input())
print((n *t)//10)
"""
"""
h = int(input())
m = int(input())
t = int(input())
ch = ((h * 60 + m + t) // 60) % 24
mn = (h * 60 + m + t) % 60
ich = imn = ""
if ch < 10:
    ich = "0"
if mn < 10:
    imn = "0"
print(ich, ch, ":", imn, mn, sep="" )
"""
"""
# x + y = k   n1 * x + n2 * y = m
k = int(input())
m = int(input())
n1 = int(input())
n2 = int(input())
x = (k * (n2 - m)) // (n2 - n1)
y = k - x
print(x, y)
"""
from math import *

def df(x):
    return 2 * x + (-15) * sin(x)

line = input().split()
x0 = float(line[0])
n = int(line[1])
k = float(line[2])
xn = x0

for i in range(1, n+1):
    xn = xn - k * df(xn)
print(round(xn, 4))
#print(f"{xn:.4f}")