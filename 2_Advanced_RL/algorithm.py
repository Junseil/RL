import sys
input = sys.stdin.readline

n = int(input())

def f(x):
    res = 0
    for i in range(1, int(x**(1/2))+1):
        if x % i == 0:
            res += i
            if i != x// i:
                res += x // i
    return res

s = 0
for x in range(1, n+1):
    s += f(x)
    
print(s)