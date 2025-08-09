import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import munch
# import matplotlib.pyplot as plt
import time

from utilities.domains import Square
from utilities.kernels import Gaussian_Kernel
from utilities.Elliptic_sweep import Elliptic


# for reproducibility
np.random.seed(2333)

# sweep
epoch_test = 10
grid_sz = 64
ell_grid = jnp.logspace(-2, 1, grid_sz)

cfg = munch.munchify({
    'M': 1200,
    'M_Omega': 900,
    'valid_M': 1200,
    'valid_M_Omega': 900,
    'learning_epoch': 50,
    'learning_tol': 1e-6,
    'learning_lr': 1e-2,
    'learning_method': "adam",
    'reg': 1,
    'alpha': 1,
    "batch_size": 200,
    'm': 3,
    'nugget': 1e-10, # nugget = 1e-12 may not give robust results
    'epoch': 30,    #epoches for training latent variables
    'lr': 1,         #learning rate for training latent variables
    'tol' : 10 ** (-9),  #tolerance for training latent variables
})

domain = Square(0, 1, 0, 1)

kernel_generator = lambda: Gaussian_Kernel()

init_params = 2.0 * jnp.ones(1)

eq = Elliptic(kernel_generator, domain)
eq.sampling(cfg)

# Perform sweep
start = time.time()
ufun, param_hist, loss_hist, sweep_history = eq.trainGN(init_params, cfg, VERBOSE_RETURN=True, my_grid=ell_grid)
print("Optimization time elapsed (sec):", time.time() - start)

print("End main sweep.")

# =============================================================================
# # Test GP-method using optimal lengthscale
# cfg['epoch'] = epoch_test
# ufun_optimal = eq.testGN(param_hist[-1], cfg)
# =============================================================================
