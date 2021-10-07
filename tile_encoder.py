import functools
import itertools
from typing import Optional, Sequence

import jax
import jax.numpy as jnp


class Bandit:
    """
    Imitating multi-armed bandit with 3-dimentional input and simple quadratic reward with normal noise
    """
    def __init__(self, offsets: Sequence = (5,4,8), std: int = 5):
        self.reward_fn = lambda x: sum(-.5*(x_i-offset_i)**2 for x_i, offset_i in zip(x, offsets))
        self.std = std

    def __call__(self, key: jax.random.PRNGKey, x: Sequence):
        return self.reward_fn(x) + jax.random.normal(key, shape=tuple())*self.std


def encode(x, offsets, l, r, widths, num_tile_layers):
    """
    encode multidimentional input with tile encoding 
    """
    x = jnp.tile(x, (num_tile_layers,1))
    offsets = jnp.tile(offsets, (num_tile_layers,1)) * jnp.arange(num_tile_layers).reshape(-1,1)
    widths = jnp.tile(widths, (num_tile_layers,1))
    return jnp.clip((x-offsets)//widths, a_min=l, a_max=r)

@jax.jit
def calculate_expectation_and_n(encoded_x, w, N):
    """
    Calculate expected reward and number N for UCB using weights (w, N)
    """
    value = 0
    n = 0
    for i, tile_numbers_i in enumerate(encoded_x):
        index = (i,) + tuple(tile_numbers_i)
        value += w[index]
        n += N[index]
    return value/len(encoded_x), n

@functools.partial(jax.vmap, in_axes=(0, None, None))
def calculate_expectation_and_n_vmap(encoded_x, w, N):
    return calculate_expectation_and_n(encoded_x, w, N)

def get_scores(points, w, N):
    values, Ns = calculate_expectation_and_n_vmap(points, w, N)
    mu_V = jnp.mean(values)
    sigma_V = jnp.std(values)

    scores = (values-mu_V)/sigma_V + jnp.sqrt(jnp.log(1+jnp.sum(Ns))/(1+Ns))
    return scores
    
def select_top_n(scores, n, grid):
    return grid[jnp.argsort(scores)][-n:]

class TileEncoder:
    """
    Class that encodes miltidimentional input feature x using tile encoding* and learns 
    weights of the tiles using output from Multi-Armed-Bandit simulator or
    from more complex non-stationary MAB, as used in GDI paper **, using UCB ***
    
    * - for more information check (Stutton, Barto, Reinforcement Learning: An Introduction, page 236)
    ** - Generalized Policy Iteration https://arxiv.org/abs/2106.06232
    *** - Upper Confidence Bound, check (Stutton, Barto, Reinforcement Learning: An Introduction, page 41)

    key - PRNGKey
    widths - widths of tiles for each dimention in x = (z_i, y_j, q_k, ...)
    offsets - offset of each tiles for each dimention in x = (z_i, y_j, q_k, ...)
    ranges - sequence of bounds for each dimention in x = (z_i, y_j, q_k, ...)
    grid - lists (z, y, q, ...)  of possible instances for each dimention in x = (z_i, y_j, q_k, ...)
    lr - learning rate for weights update
    n - number of top actions to return from get_n_actions() method
    l - left bounds for each dimention in x = (z_i, y_j, q_k, ...)
    r - right bounds for each blah blah
    """
    def __init__(self, 
                    key: jax.random.PRNGKey, 
                    widths: Sequence[int], 
                    offsets: Sequence[int], 
                    ranges: Sequence[Sequence[int]], 
                    grid: Sequence[Sequence[int]], 
                    lr: float =.01, 
                    n: int = 1, 
                    l: Optional[Sequence[int]] = None, 
                    r: Optional[Sequence[int]] = None):
        self.widths = widths
        self.offsets = offsets
        self.ranges = ranges
        self.num_tile_layers = 1
        for w, o in zip(widths, offsets):
            n_layers = jnp.lcm(o, w)//o
            self.num_tile_layers = max(self.num_tile_layers, n_layers)
        self.l = l if l else tuple(range_l_i - offset_i*self.num_tile_layers for (range_l_i, range_r_i), offset_i in zip(self.ranges, self.offsets))
        self.r = r if r else tuple(range_r_i + offset_i*self.num_tile_layers for (range_l_i, range_r_i), offset_i in zip(self.ranges, self.offsets))
        self.w = jax.random.normal(key=key, shape=((self.num_tile_layers,) + tuple((r_i - l_i)//w_i for r_i, l_i, w_i in zip(self.r, self.l, self.widths))))
        self.N = jnp.zeros((self.num_tile_layers,) + tuple((r_i - l_i)//w_i for r_i, l_i, w_i in zip(self.r, self.l, self.widths)))
        self.lr = lr
        self.grid = grid
        self.dots = jnp.array(list(itertools.product(*grid)))
        self.encode_fn = jax.jit(functools.partial(encode, offsets=jnp.array(self.offsets),
                                                    l=jnp.array(self.l),
                                                    r=jnp.array(self.r),
                                                    widths=jnp.array(self.widths),
                                                    num_tile_layers=self.num_tile_layers))
        self.encode_vmap = jax.vmap(self.encode_fn)
        self.dots_encoded = self.encode_vmap(self.dots)
        self.get_scores_fn = jax.jit(functools.partial(get_scores, points=self.dots_encoded))
        self.select_fn = jax.jit(functools.partial(select_top_n, n=n, grid=jnp.array(self.dots)))

    def update(self, x: Sequence[int], g: float):
        """
        Updating class using output g of input x = (z_i, y_j, q_k, ...)
        """
        encoded_x = self.encode_fn(x)
        self.w, self.N = self._update(encoded_x, g, self.w, self.N, self.lr)

    @staticmethod
    @functools.partial(jax.jit, static_argnums=(4,))
    def _update(encoded_x, g, w, N, lr):
        for i, tile_numbers_i in enumerate(encoded_x):
            index = (i,) + tuple(tile_numbers_i)
            w = w.at[index].add(lr * (g - calculate_expectation_and_n(encoded_x, w=w, N=N)[0]))
            N = N.at[index].add(1)
        return w, N

    def get_scores(self):
        """
        Get scores of all possible combinations of features in input x = (z_i, y_j, q_k, ...)
        sorted in order itertools.product(z_list, y_list, q_list)
        """
        return self.get_scores_fn(w=self.w, N=self.N)

    def get_n_actions(self):
        """
        returns top n actions x based on scores that get_scores method returns
        """
        return self.select_fn(self.get_scores_fn(w=self.w, N=self.N))
