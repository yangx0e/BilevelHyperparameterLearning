import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import munch
import matplotlib.pyplot as plt
import time
import scipy.io as sio
from matplotlib import colormaps

from utilities.domains import TimeDependentSquare
from utilities.kernels import Anisotropic_Periodic_Gaussian_kernel
from utilities.Schrodinger import Schrodinger


# for reproducibility
# np.random.seed(6666) # seed for training
np.random.seed(236216) # seed for testing



cfg = munch.munchify({
    'M' : 2500,
    'M_Omega' : 2000,
    'valid_M': 1700,
    'valid_M_Omega': 1500,
    'learning_epoch': 0,
    'learning_tol': 1e-5,
    'learning_lr': 1e-2,
    'sigma': jnp.array([1.0, 1.0]),
    # "sigma": jnp.array([0.2, 0.2]),
    'nugget': 1e-10,
    "batch_size": 200, 
    'epoch': 30,
    'lr': 1,
    'tol' : 10**(-7),
})

domain = TimeDependentSquare(0, 1, -5, 5)

kernel_generator = lambda: Anisotropic_Periodic_Gaussian_kernel(p=10)


eq = Schrodinger(kernel_generator, domain)
# eq.sampling(cfg)
eq.sampling_test(cfg)

init_params = [cfg.sigma, cfg.sigma]
init_params = [
    jnp.array([0.2461, 0.1455]),
    jnp.array([0.2476, 0.1340]),
] # from 1.0
# init_params = [
#     jnp.array([0.2451, 0.1457]),
#     jnp.array([0.2470, 0.1342]),
# ] # from 0.5
# init_params = [
#     jnp.array([0.2493, 0.1450]),
#     jnp.array([0.2494, 0.1338]),
# ] # from 0.2

print("Initial guess:")
print(init_params)
print("Number of validation points:")
print(cfg.valid_M, cfg.valid_M_Omega)


######## set the random seed two ########
# np.random.seed(89192)
t0 = time.time()
# u_fn, v_fn = eq.trainGN(init_params, cfg)
u_fn, v_fn = eq.testGN(init_params, cfg)
print("Elapsed: ", time.time()-t0)


################# Test #################
tt = jnp.linspace(0, 1, 101)
xx = jnp.linspace(-5, 5, 129)[:-1]
TT, XX = jnp.meshgrid(tt, xx)

X_test = jnp.concatenate([TT.reshape([-1, 1]), XX.reshape([-1, 1])], axis=1)
u_pred = jax.vmap(u_fn)(X_test)
v_pred = jax.vmap(v_fn)(X_test)

u_pred = u_pred.reshape([128, 101])
v_pred = v_pred.reshape([128, 101])
xx_test = XX.copy()
tt_test = TT.copy()

xx_test = np.array(xx_test)
tt_test = np.array(tt_test)
u_pred = np.array(u_pred)
v_pred = np.array(v_pred)


data = sio.loadmat("./data/data_schrodinger_v2.mat")
u_ref = data["us"].T
v_ref = data["vs"].T
print(u_ref.shape, v_ref.shape)
print(u_pred.shape, v_pred.shape)

plt.figure()
plt.pcolormesh(
    tt_test, 
    xx_test, 
    u_pred,
    cmap=colormaps["jet"],
)
plt.xlabel("$t$")
plt.ylabel("$x$")
# plt.axis("equal")
plt.title("$u$ (GP)")
plt.colorbar()
plt.savefig("./figures/u_gp.png")


plt.figure()
plt.pcolormesh(
    tt_test, 
    xx_test, 
    np.abs(u_ref - u_pred),
    cmap=colormaps["jet"],
)
plt.xlabel("$t$")
plt.ylabel("$x$")
# plt.axis("equal")
plt.title("Absolute error of $u$")
plt.colorbar()
plt.savefig("./figures/u_err.png")


plt.figure()
plt.pcolormesh(
    tt_test, 
    xx_test, 
    v_pred,
    cmap=colormaps["jet"],
)
plt.xlabel("$t$")
plt.ylabel("$x$")
# plt.axis("equal")
plt.title("$v$ (GP)")
plt.colorbar()
plt.savefig("./figures/v_gp.png")

plt.figure()
plt.pcolormesh(
    tt_test, 
    xx_test, 
    np.abs(v_ref - v_pred),
    cmap=colormaps["jet"],
)
plt.xlabel("$t$")
plt.ylabel("$x$")
# plt.axis("equal")
plt.title("Absolute error of $v$")
plt.colorbar()
plt.savefig("./figures/v_error.png")

h_ref = np.sqrt(u_ref**2 + v_ref**2)
h_pred = np.sqrt(u_pred**2 + v_pred**2)

sio.savemat(
    "./outputs/results",
    {
        "tt": tt_test, "xx": xx_test,
        "h_ref": h_ref, "h_pred": h_pred,
        "samples": eq.samples,
    }
)


h_ref = np.sqrt(u_ref**2 + v_ref**2)
h_pred = np.sqrt(u_pred**2 + v_pred**2)

diff = np.abs(h_ref - h_pred)
err = np.max(diff)
print("L infty error: ", err)
err = np.sqrt(np.mean(diff**2))
print("L2 error: ", err)

print("End main.")
