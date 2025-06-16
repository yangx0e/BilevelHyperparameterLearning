import os
import sys
parent_directory = os.path.abspath('..')
sys.path.append(parent_directory)


from utilities.kernels import *
from gp.NonlinearEllipticGN_Opt import *
from utilities.domains import *
import jax.numpy as jnp
from jax import config
from jax import vmap
import munch
import time
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

config.update("jax_enable_x64", True)
jnp.set_printoptions(precision=20)

cfg = munch.munchify({
    'M': 1200,
    'M_Omega': 900,
    'valid_M': 1000,
    'valid_M_Omega': 800,
    'learning_epoch': 200,
    'learning_tol': 1e-5,
    'learning_lr': 1e-3,
    'learning_method': "adam",
    'reg': 1,
    'alpha': 1,
    'm': 3,
    'ls' : 1.0, #1, #0.15142857142857144, #0.025489649819590007, #1,
    'lbda': 1e-6,
    'nugget': 1e-10,
    'epoch': 15,    #epoches for training latent variables
    'lr': 1,         #learning rate for training latent variables
    'tol' : 10 ** (-9),  #tolerance for training latent variables
})

SMALL_SIZE = 24
MEDIUM_SIZE = 28
BIGGER_SIZE = 26

plt.rcParams['axes.labelpad'] = 10
#plt.rcParams["figure.figsize"] = (8, 6)
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=21)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

domain = Square(0, 1, 0, 1)

# kernel_generator = lambda params: Gaussian_Kernel(params)
kernel_generator = lambda params: Kernel_Wrapper2D(Mild_Kernel(params))
params_reg_func = lambda params: 0 #jnp.sum(jnp.abs(params[:4]))

init_params = jnp.concatenate((jnp.ones(4), jnp.ones(6), jnp.array([jnp.log(jnp.exp(cfg.nugget) - 1)])))
# init_params = jnp.concatenate((jnp.ones(4), jnp.ones(6)))
init_params = init_params.at[3].set(0)
# init_params = cfg.ls

eq = NonlinearElliptic_GN(kernel_generator, domain, cfg.alpha, cfg.m)
eq.sampling(cfg)

#domain.random_seed(0)

start = time.time()
ufun = eq.trainGN(init_params, cfg, params_reg_func)
print("Optimization time elapsed (sec):", time.time() - start)

# GP interpolation and test accuracy
N_pts = 60
xx = jnp.linspace(0, 1, N_pts)
yy = jnp.linspace(0, 1, N_pts)
XX, YY = jnp.meshgrid(xx, yy)
X_test = jnp.concatenate((XX.reshape(-1, 1), YY.reshape(-1, 1)), axis=1)

test_truth = vmap(eq.u)(X_test[:, 0], X_test[:, 1])

print("start to resample")
u_r = vmap(ufun)(X_test)
print("finish resampling")
all_errors = jnp.abs(test_truth - u_r)
print("The final error is {}".format(jnp.max(all_errors)))

#############################################################plots########################################
################################plot samples################################
save = False
filename = "./figures/gp/nonlinear_elliptic_forward/gp_sample_points.png"
fig = plt.figure()
ax = fig.add_subplot(111)
int_data = ax.scatter(eq.samples[:eq.M_Omega, 0], eq.samples[:eq.M_Omega, 1], marker="x",
                      label='Interior nodes')
bd_data = ax.scatter(eq.samples[eq.M_Omega:, 0], eq.samples[eq.M_Omega:, 1], marker="x",
                     label='Boundary nodes')
int_data.set_clip_on(False)
bd_data.set_clip_on(False)
ax.legend(loc="upper right")
plt.title('Sample points')
if save:
    plt.savefig(filename, bbox_inches='tight')

################################plot loss history################################
# filename = "./figures/gp/nonlinear_elliptic_forward/gp_loss_history.png"
# fig = plt.figure()
# plt.plot(jnp.arange(len(eq.loss_hist)), eq.loss_hist)
# plt.yscale("log")
# plt.title('Loss history')
# plt.xlabel('The number of iterations')
# fig.tight_layout()
# if save:
#     plt.savefig(filename, bbox_inches='tight')

################################plot loss history################################
filename = "./figures/gp/nonlinear_elliptic_forward/gp_errors.png"
fmt = ticker.ScalarFormatter(useMathText=True)
fmt.set_powerlimits((0, 0))
fig = plt.figure()
ax = fig.add_subplot(111)
err_contourf = ax.contourf(XX, YY, all_errors.reshape(XX.shape), 50, cmap=plt.cm.coolwarm)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Contour of errors')
fig.colorbar(err_contourf, format=fmt)
if save:
    plt.savefig(filename, bbox_inches='tight')

###############################plot u############################################
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
p = ax.plot_surface(XX, YY, jnp.reshape(u_r, XX.shape), rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel(r'$x_1$', labelpad=15)
ax.set_ylabel(r'$x_2$', labelpad=15)
ax.set_zlabel(r'$m_{GP}$', labelpad=20)
# ax.set_zticks([0.96, 1.00, 1.04])
# ax.set_xticks([0.00, 0.50, 1.00])
# ax.set_yticks([0.00, 0.50, 1.00])
ax.tick_params(axis='z', which='major', pad=10)
#cb = fig.colorbar(p, shrink=0.5)
fig.tight_layout()

##############################################################
# save = False
# filename = ""
# fig = plt.figure()
# plt.plot(jnp.arange(len(eq.elbos_hist)), eq.elbos_hist)
# #plt.yscale("log")
# plt.title('Loss history')
# plt.xlabel('Num. of coordinate descent')
# fig.tight_layout()
# if save:
#     plt.savefig(filename, bbox_inches='tight')

# print(eq.ls)
# print(eq.elbos_hist)

plt.show()