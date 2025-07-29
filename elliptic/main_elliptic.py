import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import munch
import matplotlib.pyplot as plt
import time

from utilities.domains import Square
from utilities.kernels import Gaussian_Kernel
from utilities.Elliptic import Elliptic



# for reproducibility
np.random.seed(2333)


cfg = munch.munchify({
    'M': 1200,
    'M_Omega': 900,
    'valid_M': 1200,
    'valid_M_Omega': 900,
    'learning_epoch': 50,
    'learning_tol': 1e-5,
    'learning_lr': 1e-2,
    'learning_method': "adam",
    'reg': 1,
    'alpha': 1,
    "batch_size": 200,
    'm': 3,
    'nugget': 1e-12, # nugget = 1e-12 may not give robust results
    'epoch': 30,    #epoches for training latent variables
    'lr': 1,         #learning rate for training latent variables
    'tol' : 10 ** (-9),  #tolerance for training latent variables
})

domain = Square(0, 1, 0, 1)

kernel_generator = lambda: Gaussian_Kernel()

# init_params = jnp.concatenate((jnp.ones(4), jnp.ones(6), jnp.array([jnp.log(jnp.exp(cfg.nugget) - 1)])))
init_params = 0.1 * jnp.ones(1)
# init_params = 0.2007 * jnp.ones(1)

eq = Elliptic(kernel_generator, domain)
eq.sampling(cfg)
# eq.sampling_test()

#domain.random_seed(0)

start = time.time()
ufun = eq.trainGN(init_params, cfg)
# ufun = eq.testGN(init_params, cfg)
print("Optimization time elapsed (sec):", time.time() - start)

print("End main.")