import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import munch
import matplotlib.pyplot as plt
import time

from utilities.domains import Square
from utilities.NNKernels import LengthScaleNetwork, KernelGenerator1D
from utilities.Eikonal import Eikonal


# for reproducibility
np.random.seed(6667) 

np.set_printoptions(precision=20)


cfg = munch.munchify({
    'M' : 1200,
    'M_Omega' : 900,
    'valid_M': 1200,
    'valid_M_Omega': 900,
    "batch_size": 200,
    'learning_epoch': 50,
    'learning_tol': 1e-5,
    'learning_lr': 1e-3,
    'learning_method': "adam",
    'nugget': 1e-8,
    'epoch': 30,
    'tol' : 1e-6,
    'save': False,
})

domain = Square(0, 1, 0, 1)
nn = LengthScaleNetwork(
    layer_sizes=[2, 50, 50, 50, 1],
)
kern_gen = KernelGenerator1D(nn)
key      = jax.random.PRNGKey(0)
params0  = kern_gen.create_initial_params(key)


def count_params(params):
    """
    Count total number of scalar parameters in a JAX PyTree,
    counting Python ints/floats as 1 and DeviceArrays by their .size.
    """
    leaves, _ = jax.tree_util.tree_flatten(params)
    total = 0
    for leaf in leaves:
        # jnp arrays (and numpy arrays) have a .size
        if hasattr(leaf, 'size'):
            total += int(leaf.size)
        # Python scalars (float/int) count as 1
        else:
            total += 1
    return total
total = count_params(params0)
print(f"Total parameters in feature map: {total}")


# eq = Burgers_GP(kern_gen, domain, cfg.alpha, cfg.nu)
eq = Eikonal(kern_gen, domain, nn)
eq.sampling(cfg)

t0 = time.time()
u_fit = eq.trainGN(params0, cfg)
t1 = time.time()
print("Elapsed time for training: ", t1 - t0)

