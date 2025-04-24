import math

def idempotents(n):
    potents = []
    for i in range(0,n):
        if i%n == (i**2)%n:
            potents.append(i)
    return potents

print(idempotents(14))


