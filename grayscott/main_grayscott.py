import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import munch
import matplotlib.pyplot as plt
import time
import scipy.io as sio
from matplotlib import colormaps
import matplotlib as mpl


mpl.rcParams.update({
    # 'figure.figsize':   (4, 4),
    'figure.dpi':       300,
    'font.size':        14,   # default text size
    'axes.titlesize':   16,   # title
    'axes.labelsize':   16,   # x/y labels
    'xtick.labelsize':  14,   # tick labels
    'ytick.labelsize':  14,
    'legend.fontsize':  14,
})


from utilities.domains import TimeDependentSquare
from utilities.kernels import Anisotropic_Gaussian_kernel
from utilities.GrayScott import GrayScott

# seed for reproducibility
np.random.seed(5571) # for training
# np.random.seed(6653) # for testing


cfg = munch.munchify({
    'M' : 1000,
    'M_Omega' : 600,
    'valid_M': 1000,
    'valid_M_Omega': 600,
    'learning_epoch': 0,
    'learning_tol': 1e-5,
    'learning_lr': 1e-2,
    'learning_method': "adam",
    "u_sigma": jnp.array([1.0, 1.0]),
    "v_sigma": jnp.array([1.0, 1.0]),
    # "u_nugget": 1e-10,
    # "v_nugget": 1e-10,
    # "u_sigma": jnp.array([0.6539, 0.1125]),
    # "v_sigma": jnp.array([0.7347, 0.0899]),
    # "u_nugget": 1e-13,
    # "v_nugget": 1e-13,
    "nugget": 1e-10,
    'epoch': 20,
    "batch_size": 200,
    'lr': 1,
    'tol' : 10**(-6),
    'save': False,
})

domain = TimeDependentSquare(0, 1, 0, 1)
kernel_generator = lambda: Anisotropic_Gaussian_kernel()

eq = GrayScott(kernel_generator, domain)
eq.sampling(cfg)
# eq.sampling_testing(cfg)

init_params = [
    cfg.u_sigma, cfg.v_sigma,
]
# init_params = [
#     jnp.array([1.0542, 0.1173]),
#     jnp.array([1.5027, 0.1123]),
# ]

t0 = time.time()
u_fn, v_fn = eq.trainGN(init_params, cfg)
# u_fn, v_fn = eq.testGN(init_params, cfg)
print("Elapsed: ", time.time()-t0)

################# Test #################

tt = jnp.linspace(0, 1, 101)
xx = jnp.linspace(0, 1, 201)
TT, XX = jnp.meshgrid(tt, xx)
X_test = jnp.concatenate([TT.reshape([-1,1 ]), XX.reshape([-1, 1])], axis=1)
u_pred = jax.vmap(u_fn)(X_test)
v_pred = jax.vmap(v_fn)(X_test)

u_pred = u_pred.reshape([201, 101]).T
v_pred = v_pred.reshape([201, 101]).T
tt_test = TT.T
xx_test = XX.T

data = sio.loadmat("./data/data_grayscott.mat")
u_ref = data["u1"]
v_ref = data["u2"]


plt.figure()
plt.pcolormesh(
    xx_test, 
    tt_test, 
    u_pred,
    cmap=colormaps["jet"],
)
plt.xlabel("$x$")
plt.ylabel("$t$")
plt.title("$u$ (GP)")
plt.colorbar()
plt.tight_layout()
plt.savefig("./figures/u_gp.png", bbox_inches='tight')

plt.figure()
plt.pcolormesh(
    xx_test, 
    tt_test, 
    np.abs(u_ref-u_pred),
    cmap=colormaps["jet"],
)
plt.xlabel("$x$")
plt.ylabel("$t$")
plt.title("Absolute error of $u$")
plt.plot(eq.samples[:cfg.M_Omega, 1], eq.samples[:cfg.M_Omega, 0], "wx", label="Interior points")
plt.plot(eq.samples[cfg.M_Omega:, 1], eq.samples[cfg.M_Omega:, 0], "yx", label="Boundary points")
plt.colorbar()
plt.tight_layout()
plt.savefig("./figures/u_err.png", bbox_inches='tight')


plt.figure()
plt.pcolormesh(
    xx_test, 
    tt_test, 
    v_pred,
    cmap=colormaps["jet"],
)
plt.xlabel("$x$")
plt.ylabel("$t$")
plt.title("$v$ (GP)")
plt.colorbar()
plt.tight_layout()
plt.savefig("./figures/v_gp.png", bbox_inches='tight')

plt.figure()
plt.pcolormesh(
    xx_test, 
    tt_test, 
    np.abs(v_ref-v_pred),
    cmap=colormaps["jet"],
)
plt.xlabel("$x$")
plt.ylabel("$t$")
plt.title("Absolute error of $v$")
plt.plot(eq.samples[:cfg.M_Omega, 1], eq.samples[:cfg.M_Omega, 0], "wx", label="Interior points")
plt.plot(eq.samples[cfg.M_Omega:, 1], eq.samples[cfg.M_Omega:, 0], "yx", label="Boundary points")
plt.colorbar()
plt.tight_layout()
plt.savefig("./figures/v_err.png", bbox_inches='tight')

diff = np.abs(u_ref - u_pred)
err_u = np.sqrt(np.mean(diff**2))
print("L2 error of u: ", err_u)
err_u = np.max(diff)
print("L infty error of u: ", err_u)

diff = np.abs(v_ref - v_pred)
err_v = np.sqrt(np.mean(diff**2))
print("L2 error of v: ", err_v)
err_v = np.max(diff)
print("L infty error of v: ", err_v)
