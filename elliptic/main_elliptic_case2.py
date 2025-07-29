import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import munch
import matplotlib.pyplot as plt
import time

from utilities.domains import Square
from utilities.kernels import Kernel_Wrapper2D, Mild_Kernel
from utilities.Elliptic_case2 import Elliptic


# for reproducibility
np.random.seed(7776)


cfg = munch.munchify({
    'M': 1200,
    'M_Omega': 900,
    'valid_M': 1200,
    'valid_M_Omega': 900,
    'learning_epoch': 0,
    'learning_tol': 1e-6,
    'learning_lr': 1e-2,
    'learning_method': "adam",
    'reg': 1,
    'alpha': 1,
    "batch_size": 200,
    'm': 3,
    'nugget': 1e-10,
    'epoch': 200,    #epoches for training latent variables
    'lr': 1,         #learning rate for training latent variables
    'tol' : 10 ** (-9),  #tolerance for training latent variables
})

domain = Square(0, 1, 0, 1)

# kernel_generator = lambda params: Gaussian_Kernel(params)
kernel_generator = lambda: Kernel_Wrapper2D(Mild_Kernel())

# init_params = jnp.concatenate((jnp.ones(4), jnp.ones(6), jnp.array([jnp.log(jnp.exp(cfg.nugget) - 1)])))
init_params = jnp.array([1.0, 1.0, 1.0, 1.0])
# init_params = jnp.array([2.78989999, 0.1989443, 6.96291839, 11.20965657])

eq = Elliptic(kernel_generator, domain)
# eq.sampling(cfg)
eq.sampling_test()

#domain.random_seed(0)

start = time.time()
# ufun = eq.trainGN(init_params, cfg)
ufun = eq.testGN(init_params, cfg)
print("Optimization time elapsed (sec):", time.time() - start)

print("End main.")