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
import scipy.io as sio


from utilities.NNKernels import LengthScaleNetwork2D, KernelGenerator
from utilities.BurgersTE_test import Burgers_GP
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
    'learning_epoch': 0,
    'learning_tol': 1e-5,
    'learning_lr': 1e-3,
    'learning_method': "adam",
    'nu': 0.02,
    'nugget': 1e-10,
    'epoch': 10,
    'tol' : 10 ** (-6),
    'save': False,
})

domain = TimeDependentSquare(0, 1, -1, 1)
nn = LengthScaleNetwork2D(
    layer_sizes=[2, 50, 50, 2],
)
kern_gen = KernelGenerator(nn)
key      = jax.random.PRNGKey(0)
params  = kern_gen.create_initial_params(key)


from flax.serialization import from_bytes

with open("./checkpoints/model_20", "rb") as f:
    restored_params = from_bytes(params, f.read())


# eq = Burgers_GP(kern_gen, domain, cfg.alpha, cfg.nu)
eq = Burgers_GP(kern_gen, domain, cfg.nu, nn)
eq.sampling(cfg)

t0 = time.time()
u_fit = eq.trainGN(restored_params, cfg)
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

u_ref = jax.vmap(u_truth)(X_test[:, 0], X_test[:, 1])

print("start to resample")
u_pred = jax.vmap(u_fit)(X_test)
print("finish resampling")
all_errors = np.abs(u_pred - u_ref)
print("The final error is {}".format(np.max(all_errors)))

diff = u_pred - u_ref
err = jnp.sqrt(jnp.mean(diff**2))
print("The L2 error: ", err)


sio.savemat(
    "./outputs/results_1.mat",
    {
        "XX": XX, "YY": YY,
        "u_ref": u_ref, "u_pred": u_pred,
    }
)
