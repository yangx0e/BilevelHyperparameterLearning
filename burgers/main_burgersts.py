import os
import sys
import jax

jax.config.update("jax_enable_x64", True)
parent_directory = os.path.abspath('..')
sys.path.append(parent_directory)

import jax.numpy as jnp
import numpy as np
import munch
import matplotlib.pyplot as plt
import time


from utilities.NNKernels import LengthScaleNetwork2D, KernelGenerator
from utilities.BurgersTE import Burgers_GP
from utilities.domains import TimeDependentSquare



np.random.seed(35819)
# seed = np.random.randint(0, 2**10-1)
# seed = 648
# print(seed)
# np.random.seed(seed)

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
    'nu': 0.02,
    'nugget': 1e-10,
    'epoch': 20,
    'tol' : 10 ** (-6),
    'save': False,
})

domain = TimeDependentSquare(0, 1, -1, 1)
nn = LengthScaleNetwork2D(
    layer_sizes=[2, 50, 50, 2],
)
kern_gen = KernelGenerator(nn)
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
eq = Burgers_GP(kern_gen, domain, cfg.nu, nn)
eq.sampling(cfg)

t0 = time.time()
u_fit = eq.trainGN(params0, cfg)
t1 = time.time()
print("Elapsed time for training: ", t1 - t0)

# GP interpolation and test accuracy
[Gauss_pts, weights] = np.polynomial.hermite.hermgauss(80)
def u_truth(x1, x2):
    temp = x2-jnp.sqrt(4*cfg.nu*x1)*Gauss_pts
    val1 = weights * jnp.sin(jnp.pi*temp) * jnp.exp(-jnp.cos(jnp.pi*temp)/(2*jnp.pi*cfg.nu))
    val2 = weights * jnp.exp(-jnp.cos(jnp.pi*temp)/(2*jnp.pi*cfg.nu))
    return -jnp.sum(val1)/jnp.sum(val2)

N_pts = 60
xx = jnp.linspace(0, 1, N_pts)
yy = jnp.linspace(-1, 1, N_pts)
XX, YY = jnp.meshgrid(xx, yy)
X_test = jnp.concatenate((XX.reshape(-1,1), YY.reshape(-1,1)), axis=1)

test_truth = jax.vmap(u_truth)(X_test[:, 0], X_test[:, 1])

print("start to resample")
u_r = jax.vmap(u_fit)(X_test)
print("finish resampling")
all_errors = np.abs(test_truth - u_r)
print("The final error is {}".format(np.max(all_errors)))

# diff = test_truth - u_r
# err = jnp.sum(diff**2) / jnp.sum(test_truth**2)
# print("The relative L2 error: ", jnp.sqrt(err))

diff = test_truth - u_r
err = jnp.sqrt(jnp.mean(diff**2))
print("The relative L2 error: ", err)