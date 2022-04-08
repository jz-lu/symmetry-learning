from math import factorial
import random as rand
from copy import deepcopy

def random_insertions(n, deck):
    deck = deepcopy(deck)
    sz = len(deck)
    for _ in range(n):
        card = deck.pop(0)
        deck.insert(rand.randint(0, sz-1), card)
    return deck

from itertools import permutations as perm
deck = list(range(1, 1+8))
perms = list(perm(deck))

def trials(m, n):
    freqs = dict.fromkeys(perms, 0)
    for _ in range(m):
        shuffled = random_insertions(n, deck)
        freqs[tuple(shuffled)] += 1
    return freqs

import matplotlib.pyplot as plt
m = int(5E5)
for n in [5, 10, 15, 20]:
    print(f"n = {n}", flush=True)
    freqs = list(trials(m, n).values())
    plt.plot(range(factorial(8)), freqs, label=fr'$n = {n}$')

plt.legend()
plt.show()