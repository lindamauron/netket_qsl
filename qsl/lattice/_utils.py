import jax.numpy as jnp
from typing import Tuple


def neighbors(i:int) -> Tuple[int, int]:
    '''
    Computes the two nearest neighbors of site i s.t. k is the largest
    the order does not matter for P and Q, but for R yes
    '''
    t = jnp.floor(i/3).astype(int) # the triangle index
    j = 3*t + (i+1)%3
    k = 3*t + (i+2)%3

    return j,k