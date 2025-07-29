import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import munch
import matplotlib.pyplot as plt
import time
from matplotlib import colormaps

from utilities.domains import Square
from utilities.NNKernels import LengthScaleNetwork, KernelGenerator1D
from utilities.Eikonal_test import Eikonal


# for reproducibility
np.random.seed(6667)


np.set_printoptions(precision=20)


cfg = munch.munchify({
    'M' : 1200,
    'M_Omega' : 900,
    'valid_M' : 1200,
    'valid_M_Omega' : 900,
    "batch_size": -1,
    'learning_epoch': -1,
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
    layer_sizes=[2, 50, 50, 1],
)
kern_gen = KernelGenerator1D(nn)
key = jax.random.PRNGKey(0)
_inputs = jnp.ones(shape=[3, 2])
params = nn.init(key, _inputs)


# eq = Burgers_GP(kern_gen, domain, cfg.alpha, cfg.nu)
eq = Eikonal(kern_gen, domain, nn)
eq.sampling(cfg)

t0 = time.time()
u_fit = eq.trainGN(params, cfg)
t1 = time.time()
print("Elapsed time for training: ", t1 - t0)


##### Make prediction #####
import scipy.io as sio
data = sio.loadmat("./data/eikonal_reference_01.mat")
XX = jnp.array(data["XX"][::10, ::10])
YY = jnp.array(data["YY"][::10, ::10])
u_ref = jnp.array(data["u_ref"][::10, ::10])
X_test = jnp.concatenate([XX.reshape(-1,1), YY.reshape(-1,1)], axis=-1)

u_pred = jax.vmap(u_fit)(X_test)
u_pred = u_pred.reshape(XX.shape)

sio.savemat(
    "./outputs/results_v6_4_ref.mat",
    {
        "XX": XX, "YY": YY,
        "u_ref": u_ref, "u_pred": u_pred,
    }
)

diff = jnp.abs(u_pred - u_ref)
print("L infinity error:", jnp.max(diff))
print("L2 error:", jnp.sqrt(jnp.mean(diff**2)))

plt.figure()
plt.pcolormesh(XX, YY, u_pred, cmap=colormaps["jet"])
plt.xlabel("x")
plt.ylabel("y")
plt.title("GP")
plt.colorbar()
plt.tight_layout()
plt.savefig("./figures/u_pred_test.png")
plt.close()

plt.figure()
plt.pcolormesh(XX, YY, u_ref, cmap=colormaps["jet"])
plt.xlabel("x")
plt.ylabel("y")
plt.title("Reference")
plt.colorbar()
plt.tight_layout()
plt.savefig("./figures/u_ref_test.png")
plt.close()

