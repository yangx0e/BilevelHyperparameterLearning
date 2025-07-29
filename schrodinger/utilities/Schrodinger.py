import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import vmap, jacfwd, value_and_grad, grad, lax
from utilities.domains import *
import jax.scipy as jsp
import optax
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import scipy.io as sio


np.set_printoptions(precision=20)


class Schrodinger(object):
    # The Schrodinger equation:
    # equation 1: u_t + 0.5 v_xx + (u^2 + v^2) v = 0
    # equation 2: v_t - 0.5 u_xx - (u^2 + v^2) u = 0
    def __init__(self, kernel_generator, domain):
        self.kernel_generator = kernel_generator
        self.domain = domain

    def sampling(self, cfg):
        # self.samples = self.domain.sampling(cfg.M, cfg.M_Omega)
        # self.M = cfg.M
        # self.M_Omega = cfg.M_Omega
        # self.M_0 = cfg.M - cfg.M_Omega
        # self.samples = jnp.array(self.samples)

        # self.samples_valid = self.domain.sampling(cfg.valid_M, cfg.valid_M_Omega)
        # self.samples_interior_valid = self.samples_valid[:cfg.valid_M_Omega]

        # t, x = jnp.split(self.samples, 2, axis=1)
        # # initial condition: cos(2pi x)/sech(10x)
        # self.u0 = 2 / jnp.cosh(x[self.M_Omega:self.M_Omega+self.M_0]) * (t[self.M_Omega:self.M_Omega+self.M_0] == 0)
        # self.v0 = jnp.zeros_like(self.u0)
        # self.u0 = self.u0.flatten()
        # self.v0 = self.v0.flatten()

        # sio.savemat(
        #     "./data/schrodinger.mat",
        #     {
        #         "samples": self.samples,
        #         "M": cfg.M, "M_Omega": cfg.M_Omega, "M_0": self.M_0,
        #         "u0": self.u0, "v0": self.v0,
        #         "samples_valid": self.samples_valid,
        #         "samples_interior_valid": self.samples_interior_valid,
        #         "valid_M": cfg.valid_M, "valid_M_Omega": cfg.valid_M_Omega,
        #     }
        # )

        data = sio.loadmat("./data/schrodinger.mat")
        self.samples = data["samples"]
        # self.samples_valid = data["samples_valid"]
        self.samples_interior_valid = data["samples_interior_valid"]
        self.M = data["M"].astype(np.int32).reshape([])
        self.M_Omega = data["M_Omega"].astype(np.int32).reshape([])
        self.M_0 = data["M_0"].astype(np.int32).reshape([])
        self.u0 = data["u0"].flatten()
        self.v0 = data["v0"].flatten()

        self.valid_M = data["valid_M"].astype(np.int32).reshape([])
        self.valid_M_Omega = data["valid_M_Omega"].astype(np.int32).reshape([])

    def sampling_test(self, cfg):
        self.samples = self.domain.sampling(cfg.M, cfg.M_Omega)
        self.M = cfg.M
        self.M_Omega = cfg.M_Omega
        self.M_0 = cfg.M - cfg.M_Omega
        self.samples = jnp.array(self.samples)

        t, x = jnp.split(self.samples, 2, axis=1)
        # initial condition: cos(2pi x)/sech(10x)
        self.u0 = 2 / jnp.cosh(x[self.M_Omega:self.M_Omega+self.M_0]) * (t[self.M_Omega:self.M_Omega+self.M_0] == 0)
        self.v0 = jnp.zeros_like(self.u0)
        self.u0 = self.u0.flatten()
        self.v0 = self.v0.flatten()

    # def sampling_test(self, cfg):
    #     data = sio.loadmat("./data/schrodinger.mat")
    #     samples = data["samples"]
    #     M = data["M"].astype(np.int32).reshape([])
    #     M_Omega = data["M_Omega"].astype(np.int32).reshape([])
        
    #     # valid_M_Omega = data["valid_M_Omega"].astype(np.int32).reshape([])
    #     # samples_interior_valid = data["samples_interior_valid"]
    #     # self.samples = jnp.concatenate(
    #     #     [samples[:M_Omega], samples_interior_valid, samples[M_Omega:]], axis=0,
    #     # )
    #     self.samples = samples
    #     self.u0 = jnp.array(data["u0"].flatten())
    #     self.v0 = jnp.array(data["v0"].flatten())
    #     # self.M = M + valid_M_Omega
    #     # self.M_Omega = M_Omega + valid_M_Omega
    #     self.M = M
    #     self.M_Omega = M_Omega
    #     self.M_0 = data["M_0"].astype(np.int32).reshape([])

    def nonlinear_residual(self, z, zk, M_Omega):
        zu, zv = jnp.split(z, 2, axis=0)
        zku, zkv = jnp.split(zk, 2, axis=0)

        ut = zu[0:M_Omega]  # u_t
        u = zu[M_Omega:2*M_Omega]  # u
        vt = zv[0:M_Omega]  # v_t
        v = zv[M_Omega:2*M_Omega]  # v

        # ukt = zku[0:M_Omega]  # uk_t
        uk = zku[M_Omega:2*M_Omega]  # uk
        # vkt = zkv[0:M_Omega]  # vk_t
        vk = zkv[M_Omega:2*M_Omega]  # vk

        # equation 1: u_t + 0.5 v_xx + (u^2 + v^2) v = 0
        # equation 2: v_t - 0.5 u_xx - (u^2 + v^2) u = 0
        # linearized equation 1: u_t + 0.5 v_xx + (uk^2 v + 2uk vk u - 2uk^2 vk + 3vk^2 v - 2vk^3)
        # linearized equation 2: v_t - 0.5 u_xx - (3uk^2 u - 2uk^3 + vk^2 u + 2uk vk v - 2vk^2 uk)
        
        uxx = 2 * (vt - (3*uk**2*u - 2*uk**3 + vk**2*u + 2*uk*vk*v - 2*vk**2*uk))
        vxx = 2 * (-ut - (uk**2*v + 2*uk*vk*u - 2*uk**2*vk + 3*vk**2*v - 2*vk**3))

        eq1 = ut + 0.5 * vxx + (u**2 + v**2) * v
        eq2 = vt - 0.5 * uxx - (u**2 + v**2) * u
        return jnp.concatenate([eq1, eq2], axis=0)
    
    def residual(self, z, zk, M_Omega):
        zu, zv = jnp.split(z, 2, axis=0)
        zku, zkv = jnp.split(zk, 2, axis=0)

        ut = zu[0:M_Omega]
        uxx = zu[M_Omega:2*M_Omega]
        u = zu[2*M_Omega:3*M_Omega]
        vt = zv[0:M_Omega]
        vxx = zv[M_Omega:2*M_Omega]
        v = zv[2*M_Omega:3*M_Omega]

        uk = zku[2*M_Omega:3*M_Omega]
        vk = zkv[2*M_Omega:3*M_Omega]

        # 0.5 uxx = vt - (3*uk**2*u - 2*uk**3 + vk**2*u + 2*uk*vk*v - 2*vk**2*uk)
        # 0.5 vxx = -ut - (uk**2*v + 2*uk*vk*u - 2*uk**2*vk + 3*vk**2*v - 2*vk**3)
        eq1 = ut + 0.5 * vxx + (uk**2*v + 2*uk*vk*u - 2*uk**2*vk + 3*vk**2*v - 2*vk**3)
        eq2 = vt - 0.5 * uxx - (3*uk**2*u - 2*uk**3 + vk**2*u + 2*uk*vk*v - 2*vk**2*uk)
        return jnp.concatenate([eq1, eq2], axis=0)
    
    def prolong(self, z, zk, M_Omega):
        zu, zv = jnp.split(z, 2, axis=0)
        zku, zkv = jnp.split(zk, 2, axis=0)

        ut = zu[0:M_Omega]  # u_t
        u = zu[M_Omega:2*M_Omega]  # u
        vt = zv[0:M_Omega]  # v_t
        v = zv[M_Omega:2*M_Omega]  # v

        uk = zku[M_Omega:2*M_Omega]  # uk
        vk = zkv[M_Omega:2*M_Omega]  # vk

        uxx = 2 * (vt - (3*uk**2*u - 2*uk**3 + vk**2*u + 2*uk*vk*v - 2*vk**2*uk))
        vxx = 2 * (-ut - (uk**2*v + 2*uk*vk*u - 2*uk**2*vk + 3*vk**2*v - 2*vk**3))
        rhs_1 = jnp.concatenate(
            [ut, uxx, u, self.u0], axis=0,
        )
        rhs_2 = jnp.concatenate(
            [vt, vxx, v, self.v0], axis=0,
        )
        return jnp.concatenate([rhs_1, rhs_2], axis=0)
    
    def gen_uv(self, params, z, zk, cfg):
        params_u, params_v = params
        u_kernel = self.kernel_generator()
        v_kernel = self.kernel_generator()

        theta_u = self.build_theta(
            u_kernel, self.samples, params_u,
        )
        theta_v = self.build_theta(
            v_kernel, self.samples, params_v,
        )
        nuggets_u = self.build_nuggets(theta_u)
        nuggets_v = self.build_nuggets(theta_v)
        gram_u = theta_u + cfg.nugget * nuggets_u
        gram_v = theta_v + cfg.nugget * nuggets_v
        cho_u = jax.scipy.linalg.cho_factor(gram_u)
        cho_v = jax.scipy.linalg.cho_factor(gram_v)

        uvz = self.prolong(z, zk, self.M_Omega)
        uz, vz = jnp.split(uvz, 2, axis=0)
        u_weights = jax.scipy.linalg.cho_solve(cho_u, uz)
        v_weights = jax.scipy.linalg.cho_solve(cho_v, vz)
        u_fn = lambda x: jnp.dot(self.KxPhi(u_kernel, x, params_u), u_weights)
        v_fn = lambda x: jnp.dot(self.KxPhi(v_kernel, x, params_v), v_weights)
        return u_fn, v_fn

    def KxPhi(self, kernel, nx, params):
        x = self.samples

        x1_v = nx[0].reshape([-1, 1])
        x2_v = nx[1].reshape([-1, 1])

        M_Omega = self.M_Omega
        M_0 = self.M_0
        kxphi = jnp.zeros([1, 3*M_Omega+M_0])
        x1_c = x[:M_Omega, 0:1]
        x2_c = x[:M_Omega, 1:2]
        x1_0 = x[M_Omega:M_Omega+M_0, 0:1]
        x2_0 = x[M_Omega:M_Omega+M_0, 1:2]
        
        # u vs ut, uxx, u
        x1_vc_l = jnp.tile(x1_v, [1, M_Omega]).flatten()
        x2_vc_l = jnp.tile(x2_v, [1, M_Omega]).flatten()
        x1_vc_r = x1_c.T.flatten()
        x2_vc_r = x2_c.T.flatten()

        # u v.s. 0
        x1_v0_l = jnp.tile(x1_v, [1, M_0]).flatten()
        x2_v0_l = jnp.tile(x2_v, [1, M_0]).flatten()
        x1_v0_r = x1_0.T.flatten()
        x2_v0_r = x2_0.T.flatten()
        
        # u vs ut
        val = vmap(
            lambda x1, x2, y1, y2: kernel.D_y1_kappa(x1, x2, y1, y2, params)
        )(x1_vc_l, x2_vc_l, x1_vc_r, x2_vc_r).reshape([1, M_Omega])
        kxphi = kxphi.at[:, 0:M_Omega].set(val)
        # u vs uxx
        val = vmap(
            lambda x1, x2, y1, y2: kernel.DD_y2_kappa(x1, x2, y1, y2, params)
        )(x1_vc_l, x2_vc_l, x1_vc_r, x2_vc_r).reshape([1, M_Omega])
        kxphi = kxphi.at[:, M_Omega:2*M_Omega].set(val)
        # u vs u
        val = vmap(
            lambda x1, x2, y1, y2: kernel.kappa(x1, x2, y1, y2, params)
        )(x1_vc_l, x2_vc_l, x1_vc_r, x2_vc_r).reshape([1, M_Omega])
        kxphi = kxphi.at[:, 2*M_Omega:3*M_Omega].set(val)
        # u vs u0
        val = vmap(
            lambda x1, x2, y1, y2: kernel.kappa(x1, x2, y1, y2, params)
        )(x1_v0_l, x2_v0_l, x1_v0_r, x2_v0_r).reshape([1, M_0])
        kxphi = kxphi.at[:, 3*M_Omega:3*M_Omega+M_0].set(val)
        return kxphi.flatten()
    
    def build_theta(self, kernel, x, params):
        M_Omega = self.M_Omega
        M_0 = self.M_0
        theta = jnp.zeros([3*M_Omega+M_0, 3*M_Omega+M_0])
        x1_c = x[:M_Omega, 0:1]
        x2_c = x[:M_Omega, 1:2]
        x1_0 = x[M_Omega:M_Omega+M_0, 0:1]
        x2_0 = x[M_Omega:M_Omega+M_0, 1:2]
        
        # ut, uxx, u, u0
        # vt, vxx, v, v0
        # int v.s. int
        x1_cc_l = jnp.tile(x1_c, [1, M_Omega]).flatten()
        x2_cc_l = jnp.tile(x2_c, [1, M_Omega]).flatten()
        x1_cc_r = jnp.tile(x1_c.T, [M_Omega, 1]).flatten()
        x2_cc_r = jnp.tile(x2_c.T, [M_Omega, 1]).flatten()

        # int v.s. 0
        x1_c0_l = jnp.tile(x1_c, [1, M_0]).flatten()
        x2_c0_l = jnp.tile(x2_c, [1, M_0]).flatten()
        x1_c0_r = jnp.tile(x1_0.T, [M_Omega, 1]).flatten()
        x2_c0_r = jnp.tile(x2_0.T, [M_Omega, 1]).flatten()

        # 0 v.s. 0
        x1_00_l = jnp.tile(x1_0, [1, M_0]).flatten()
        x2_00_l = jnp.tile(x2_0, [1, M_0]).flatten()
        x1_00_r = jnp.tile(x1_0.T, [M_0, 1]).flatten()
        x2_00_r = jnp.tile(x2_0.T, [M_0, 1]).flatten()


        #### part 1: ut v.s. the rest
        # ut vs ut
        val = vmap(
            lambda x1, x2, y1, y2: kernel.D_x1_D_y1_kappa(x1, x2, y1, y2, params)
        )(x1_cc_l, x2_cc_l, x1_cc_r, x2_cc_r).reshape([M_Omega, M_Omega])
        theta = theta.at[0:M_Omega, 0:M_Omega].set(val)
        # ut vs uxx
        val = vmap(
            lambda x1, x2, y1, y2: kernel.D_x1_DD_y2_kappa(x1, x2, y1, y2, params)
        )(x1_cc_l, x2_cc_l, x1_cc_r, x2_cc_r).reshape([M_Omega, M_Omega])
        theta = theta.at[0:M_Omega, M_Omega:2*M_Omega].set(val)
        theta = theta.at[M_Omega:2*M_Omega, 0:M_Omega].set(val.T)

        # ut vs u
        val = vmap(
            lambda x1, x2, y1, y2: kernel.D_x1_kappa(x1, x2, y1, y2, params)
        )(x1_cc_l, x2_cc_l, x1_cc_r, x2_cc_r).reshape([M_Omega, M_Omega])
        theta = theta.at[0:M_Omega, 2*M_Omega:3*M_Omega].set(val)
        theta = theta.at[2*M_Omega:3*M_Omega, 0:M_Omega].set(val.T)

        # ut vs u0
        val = vmap(
            lambda x1, x2, y1, y2: kernel.D_x1_kappa(x1, x2, y1, y2, params)
        )(x1_c0_l, x2_c0_l, x1_c0_r, x2_c0_r).reshape([M_Omega, M_0])
        theta = theta.at[0:M_Omega, 3*M_Omega:3*M_Omega+M_0].set(val)
        theta = theta.at[3*M_Omega:3*M_Omega+M_0, 0:M_Omega].set(val.T)
        
        #### part 2: uxx v.s. the rest
        # uxx vs uxx
        val = vmap(
            lambda x1, x2, y1, y2: kernel.DD_x2_DD_y2_kappa(x1, x2, y1, y2, params)
        )(x1_cc_l, x2_cc_l, x1_cc_r, x2_cc_r).reshape([M_Omega, M_Omega])
        theta = theta.at[M_Omega:2*M_Omega, M_Omega:2*M_Omega].set(val)

        # uxx vs u
        val = vmap(
            lambda x1, x2, y1, y2: kernel.DD_x2_kappa(x1, x2, y1, y2, params)
        )(x1_cc_l, x2_cc_l, x1_cc_r, x2_cc_r).reshape([M_Omega, M_Omega])
        theta = theta.at[M_Omega:2*M_Omega, 2*M_Omega:3*M_Omega].set(val)
        theta = theta.at[2*M_Omega:3*M_Omega, M_Omega:2*M_Omega].set(val.T)

        # uxx vs u0
        val = vmap(
            lambda x1, x2, y1, y2: kernel.DD_x2_kappa(x1, x2, y1, y2, params)
        )(x1_c0_l, x2_c0_l, x1_c0_r, x2_c0_r).reshape([M_Omega, M_0])
        theta = theta.at[M_Omega:2*M_Omega, 3*M_Omega:3*M_Omega+M_0].set(val)
        theta = theta.at[3*M_Omega:3*M_Omega+M_0, M_Omega:2*M_Omega].set(val.T)

        #### part 3: u v.s. the rest
        # u vs u
        val = vmap(
            lambda x1, x2, y1, y2: kernel.kappa(x1, x2, y1, y2, params)
        )(x1_cc_l, x2_cc_l, x1_cc_r, x2_cc_r).reshape([M_Omega, M_Omega])
        theta = theta.at[2*M_Omega:3*M_Omega, 2*M_Omega:3*M_Omega].set(val)

        # u vs u0
        val = vmap(
            lambda x1, x2, y1, y2: kernel.kappa(x1, x2, y1, y2, params)
        )(x1_c0_l, x2_c0_l, x1_c0_r, x2_c0_r).reshape([M_Omega, M_0])
        theta = theta.at[2*M_Omega:3*M_Omega, 3*M_Omega:3*M_Omega+M_0].set(val)
        theta = theta.at[3*M_Omega:3*M_Omega+M_0, 2*M_Omega:3*M_Omega].set(val.T)

        #### part 4: u0 v.s. the rest
        # u0 vs u0
        val = vmap(
            lambda x1, x2, y1, y2: kernel.kappa(x1, x2, y1, y2, params)
        )(x1_00_l, x2_00_l, x1_00_r, x2_00_r).reshape([M_0, M_0])
        theta = theta.at[3*M_Omega:3*M_Omega+M_0, 3*M_Omega:3*M_Omega+M_0].set(val)

        return theta
    
    def build_nuggets(self, theta):
        M_Omega = self.M_Omega
        M_0 = self.M_0

        trace11 = jnp.trace(theta[0:M_Omega, 0:M_Omega]) # ut
        trace22 = jnp.trace(theta[M_Omega:2*M_Omega, M_Omega:2*M_Omega]) # uxx
        trace33 = jnp.trace(theta[2*M_Omega:3*M_Omega, 2*M_Omega:3*M_Omega]) # u
        trace44 = jnp.trace(theta[3*M_Omega:3*M_Omega+M_0, 3*M_Omega:3*M_Omega+M_0]) # u0

        ratio = [
            trace11/(trace33+trace44), trace22/(trace33+trace44),
        ]
        r_diag = jnp.concatenate(
            [
                ratio[0] * jnp.ones([1, M_Omega]),
                ratio[1] * jnp.ones([1, M_Omega]),
                jnp.ones([1, M_Omega]),
                jnp.ones([1, M_0]),
            ], 
            axis=1,
        )
        r = jnp.diag(r_diag[0])
        return r

    def eval_ops(self, kernel, weights, nx, params):
        # TODO: add support for different kernels
        x = self.samples
        M_Omega = self.M_Omega
        M_0 = self.M_0
        x1_c = x[:M_Omega, 0:1]
        x2_c = x[:M_Omega, 1:2]
        x1_0 = x[M_Omega:M_Omega+M_0, 0:1]
        x2_0 = x[M_Omega:M_Omega+M_0, 1:2]

        M_v = nx.shape[0]
        x1_v = nx[:, 0:1] # validation data
        x2_v = nx[:, 1:2] # validation data
        
        # ut, uxx, u, u0
        # vt, vxx, v, v0
        # int v.s. int
        x1_vc_l = jnp.tile(x1_v, [1, M_Omega]).flatten()
        x2_vc_l = jnp.tile(x2_v, [1, M_Omega]).flatten()
        x1_vc_r = jnp.tile(x1_c.T, [M_v, 1]).flatten()
        x2_vc_r = jnp.tile(x2_c.T, [M_v, 1]).flatten()

        # int v.s. 0
        x1_v0_l = jnp.tile(x1_v, [1, M_0]).flatten()
        x2_v0_l = jnp.tile(x2_v, [1, M_0]).flatten()
        x1_v0_r = jnp.tile(x1_0.T, [M_v, 1]).flatten()
        x2_v0_r = jnp.tile(x2_0.T, [M_v, 1]).flatten()

        # ut vs ut, uxx, u, u0
        mtx = jnp.zeros([M_v, 3*M_Omega+M_0])
        val = vmap(
            lambda x1, x2, y1, y2: kernel.D_x1_D_y1_kappa(x1, x2, y1, y2, params)
        )(x1_vc_l, x2_vc_l, x1_vc_r, x2_vc_r).reshape([M_v, M_Omega])
        mtx = mtx.at[:, 0:M_Omega].set(val)

        val = vmap(
            lambda x1, x2, y1, y2: kernel.D_x1_DD_y2_kappa(x1, x2, y1, y2, params)
        )(x1_vc_l, x2_vc_l, x1_vc_r, x2_vc_r).reshape([M_v, M_Omega])
        mtx = mtx.at[:, M_Omega:2*M_Omega].set(val)

        val = vmap(
            lambda x1, x2, y1, y2: kernel.D_x1_kappa(x1, x2, y1, y2, params)
        )(x1_vc_l, x2_vc_l, x1_vc_r, x2_vc_r).reshape([M_v, M_Omega])
        mtx = mtx.at[:, 2*M_Omega:3*M_Omega].set(val)

        val = vmap(
            lambda x1, x2, y1, y2: kernel.D_x1_kappa(x1, x2, y1, y2, params)
        )(x1_v0_l, x2_v0_l, x1_v0_r, x2_v0_r).reshape([M_v, M_0])
        mtx = mtx.at[:, 3*M_Omega:3*M_Omega+M_0].set(val)

        ut_v = mtx.dot(weights)

        # uxx vs ut, uxx, u, u0
        mtx = jnp.zeros([M_v, 3*M_Omega+M_0])
        val = vmap(
            lambda x1, x2, y1, y2: kernel.DD_x2_D_y1_kappa(x1, x2, y1, y2, params)
        )(x1_vc_l, x2_vc_l, x1_vc_r, x2_vc_r).reshape([M_v, M_Omega])
        mtx = mtx.at[:, 0:M_Omega].set(val)

        val = vmap(
            lambda x1, x2, y1, y2: kernel.DD_x2_DD_y2_kappa(x1, x2, y1, y2, params)
        )(x1_vc_l, x2_vc_l, x1_vc_r, x2_vc_r).reshape([M_v, M_Omega])
        mtx = mtx.at[:, M_Omega:2*M_Omega].set(val)

        val = vmap(
            lambda x1, x2, y1, y2: kernel.DD_x2_kappa(x1, x2, y1, y2, params)
        )(x1_vc_l, x2_vc_l, x1_vc_r, x2_vc_r).reshape([M_v, M_Omega])
        mtx = mtx.at[:, 2*M_Omega:3*M_Omega].set(val)

        val = vmap(
            lambda x1, x2, y1, y2: kernel.DD_x2_kappa(x1, x2, y1, y2, params)
        )(x1_v0_l, x2_v0_l, x1_v0_r, x2_v0_r).reshape([M_v, M_0])
        mtx = mtx.at[:, 3*M_Omega:3*M_Omega+M_0].set(val)

        uxx_v = mtx.dot(weights)

        # u vs ut, uxx, u, u0
        mtx = jnp.zeros([M_v, 3*M_Omega+M_0])
        val = vmap(
            lambda x1, x2, y1, y2: kernel.D_y1_kappa(x1, x2, y1, y2, params)
        )(x1_vc_l, x2_vc_l, x1_vc_r, x2_vc_r).reshape([M_v, M_Omega])
        mtx = mtx.at[:, 0:M_Omega].set(val)

        val = vmap(
            lambda x1, x2, y1, y2: kernel.DD_y2_kappa(x1, x2, y1, y2, params)
        )(x1_vc_l, x2_vc_l, x1_vc_r, x2_vc_r).reshape([M_v, M_Omega])
        mtx = mtx.at[:, M_Omega:2*M_Omega].set(val)

        val = vmap(
            lambda x1, x2, y1, y2: kernel.kappa(x1, x2, y1, y2, params)
        )(x1_vc_l, x2_vc_l, x1_vc_r, x2_vc_r).reshape([M_v, M_Omega])
        mtx = mtx.at[:, 2*M_Omega:3*M_Omega].set(val)

        val = vmap(
            lambda x1, x2, y1, y2: kernel.kappa(x1, x2, y1, y2, params)
        )(x1_v0_l, x2_v0_l, x1_v0_r, x2_v0_r).reshape([M_v, M_0])
        mtx = mtx.at[:, 3*M_Omega:3*M_Omega+M_0].set(val)

        u_v = mtx.dot(weights)

        us = [ut_v, uxx_v, u_v]
        return us
    
    def trainGN(self, init_params, cfg):
        error = 1
        it = 0

        params = init_params
        uk_params, vk_params = params
        u_kernel = self.kernel_generator()
        v_kernel = self.kernel_generator()

        uk_weights = jnp.zeros(3*self.M_Omega + self.M_0)
        vk_weights = jnp.zeros(3*self.M_Omega + self.M_0)

        zk = jnp.zeros([4 * self.M_Omega, ])  # represent the values of operators at the samples

        params_history_u = [uk_params]
        params_history_v = [vk_params] 
        while  it < cfg.epoch: #error > cfg.tol and iter < cfg.epoch:
            # The optimization problem at each GN iteration is formulated as the following optimization problem
            # (Jz + y)^T\Theta^{-1}(Jz + y).
            # The minimization problem admits an explicit solution
            # z = -(J^T\Theta^{-1}J)^{-1}J^T\Theta^{-1}y

            # zk_current represents the values of operators of the functions at the previous iterations
            # evaluated at samples in the current iteration
            zk_current = zk
            zl_zeros = jnp.zeros(4*self.M_Omega)
            uk_params, vk_params = params

            y = self.prolong(zl_zeros, zk_current, self.M_Omega)
            J = jacfwd(self.prolong)(zl_zeros, zk_current, self.M_Omega)
            y1, y2 = jnp.split(y, 2, axis=0)
            J1, J2 = jnp.split(J, 2, axis=0)

            idx = np.random.choice(
                self.samples_interior_valid.shape[0],
                self.samples_interior_valid.shape[0],
                replace=False,
            )[: cfg.batch_size]
            batch_samples_interior_valid = self.samples_interior_valid[idx, :]

            uks = self.eval_ops(
                u_kernel, uk_weights, batch_samples_interior_valid, uk_params,
            )
            vks = self.eval_ops(
                v_kernel, vk_weights, batch_samples_interior_valid, vk_params,
            )
            zuk_at_valid = jnp.concatenate(uks)
            zvk_at_valid = jnp.concatenate(vks)
            zk_at_valid = jnp.concatenate([zuk_at_valid, zvk_at_valid], axis=0)
            zk_at_valid = jax.lax.stop_gradient(zk_at_valid)
            self.batch_size = cfg.batch_size


            def loss_learning_params(params, zk_at_valid, batch_samples_interior_valid):
                params_u, params_v = params
                theta_u = self.build_theta(
                    u_kernel, self.samples, params_u,
                )
                theta_v = self.build_theta(
                    v_kernel, self.samples, params_v,
                )
                nuggets_u = self.build_nuggets(theta_u)
                nuggets_v = self.build_nuggets(theta_v)
                gram_u = theta_u + cfg.nugget * nuggets_u
                gram_v = theta_v + cfg.nugget * nuggets_v
                cho_u = jax.scipy.linalg.cho_factor(gram_u)
                cho_v = jax.scipy.linalg.cho_factor(gram_v)

                A = jnp.dot(J1.T, jax.scipy.linalg.cho_solve(cho_u, J1)) + \
                    jnp.dot(J2.T, jax.scipy.linalg.cho_solve(cho_v, J2))
                b = -jnp.dot(J1.T, jax.scipy.linalg.cho_solve(cho_u, y1)) + \
                    -jnp.dot(J2.T, jax.scipy.linalg.cho_solve(cho_v, y2))
                new_z = jnp.linalg.solve(A, b)
                new_uvz = self.prolong(new_z, zk_current, self.M_Omega)
                new_uz, new_vz = jnp.split(new_uvz, 2, axis=0)

                u_weights = jax.scipy.linalg.cho_solve(cho_u, new_uz)
                v_weights = jax.scipy.linalg.cho_solve(cho_v, new_vz)

                us = self.eval_ops(u_kernel, u_weights, batch_samples_interior_valid, params_u)
                vs = self.eval_ops(v_kernel, v_weights, batch_samples_interior_valid, params_v)
                zu_at_valid = jnp.concatenate(us)
                zv_at_valid = jnp.concatenate(vs)
                z_at_valid = jnp.concatenate([zu_at_valid, zv_at_valid], axis=0)

                res = self.residual(z_at_valid, zk_at_valid, self.batch_size)

                return jnp.dot(res, res)
            

            @jax.jit
            def compute_updates(
                params, zk_at_valid, batch_samples_interior_valid, opt_state
            ):
                current_loss, grads = value_and_grad(loss_learning_params)(
                    params,
                    zk_at_valid,
                    batch_samples_interior_valid,
                )
                update, opt_state = optimizer.update(grads, opt_state, params)
                return update, opt_state, current_loss
            

            optimizer = optax.adam(learning_rate=cfg.learning_lr)
            opt_state = optimizer.init(params)

            for epoch in range(cfg.learning_epoch):

                update, opt_state, current_loss = compute_updates(
                    params,
                    zk_at_valid,
                    batch_samples_interior_valid,
                    opt_state,
                )
                params = optax.apply_updates(params, update)
                params_history_u += [params[0]]
                params_history_v += [params[1]]

                # print(f"        Learning Hyperparameters, GN Epoch {it}, Learning Hyper Epoch {epoch}, Current Loss {current_loss}, Current parameter {params}")
                print(
                    f"        Learning Hyperparameters, GN Epoch {it}, Learning Hyper Epoch {epoch}, Current Loss {current_loss}",
                    flush=True,
                )

                if current_loss <= cfg.learning_tol:
                    print(f"Stopping early at epoch {epoch} due to loss threshold")
                    break

            
            @jax.jit
            def solve(params):
                params_u, params_v = params
                theta_u = self.build_theta(
                    u_kernel, self.samples, params_u,
                )
                theta_v = self.build_theta(
                    v_kernel, self.samples, params_v,
                )
                nuggets_u = self.build_nuggets(theta_u)
                nuggets_v = self.build_nuggets(theta_v)
                gram_u = theta_u + cfg.nugget * nuggets_u
                gram_v = theta_v + cfg.nugget * nuggets_v
                cho_u = jax.scipy.linalg.cho_factor(gram_u)
                cho_v = jax.scipy.linalg.cho_factor(gram_v)

                A = jnp.dot(J1.T, jax.scipy.linalg.cho_solve(cho_u, J1)) + \
                    jnp.dot(J2.T, jax.scipy.linalg.cho_solve(cho_v, J2))
                b = -jnp.dot(J1.T, jax.scipy.linalg.cho_solve(cho_u, y1)) + \
                    -jnp.dot(J2.T, jax.scipy.linalg.cho_solve(cho_v, y2))
                new_z = jnp.linalg.solve(A, b)
                new_uvz = self.prolong(new_z, zk_current, self.M_Omega)
                new_uz, new_vz = jnp.split(new_uvz, 2, axis=0)

                u_weights = jax.scipy.linalg.cho_solve(cho_u, new_uz)
                v_weights = jax.scipy.linalg.cho_solve(cho_v, new_vz)
                res = self.nonlinear_residual(new_z, zk_current, self.M_Omega)
                # res = self.nonlinear_residual(new_z, self.M_Omega)
                error = jnp.dot(res, res)
                return error, u_weights, v_weights, new_z


            # res = self.nonlinear_residual(new_z, self.M_Omega)
            error, u_weights, v_weights, new_z = solve(params)

            it = it + 1

            print(f"GN Epoch {it}, param {params}, PDE Error {error}", flush=True)

            
            # making plots
            u_fn, v_fn = self.gen_uv(params, new_z, new_z, cfg)
            tt = jnp.linspace(0, 1, 101)
            xx = jnp.linspace(-5, 5, 129)[:-1]
            TT, XX = jnp.meshgrid(tt, xx)
            X_test = jnp.concatenate([TT.reshape([-1, 1]), XX.reshape([-1, 1])], axis=1)
            u_pred = vmap(u_fn)(X_test)
            v_pred = vmap(v_fn)(X_test)

            u_pred = u_pred.reshape([128, 101])
            v_pred = v_pred.reshape([128, 101])
            xx_test = XX.copy()
            tt_test = TT.copy()

            xx_test = np.array(xx_test)
            tt_test = np.array(tt_test)
            u_pred = np.array(u_pred)
            v_pred = np.array(v_pred)
            
            plt.figure()
            plt.pcolormesh(
                tt_test, 
                xx_test, 
                u_pred,
                cmap=colormaps["jet"],
            )
            plt.xlabel("$t$")
            plt.ylabel("$x$")
            plt.axis("equal")
            plt.title("$u$ (GP)")
            plt.savefig("./figures/u_gp.png")
            plt.close()

            plt.figure()
            plt.pcolormesh(
                tt_test, 
                xx_test, 
                v_pred,
                cmap=colormaps["jet"],
            )
            plt.xlabel("$t$")
            plt.ylabel("$x$")
            plt.axis("equal")
            plt.title("$v$ (GP)")
            plt.savefig("./figures/v_gp.png")
            plt.close()

            params = jax.lax.stop_gradient(params)
            uk_weights = jax.lax.stop_gradient(u_weights)
            vk_weights = jax.lax.stop_gradient(v_weights)
            zk = jax.lax.stop_gradient(new_z)


        u_fn, v_fn = self.gen_uv(params, zk, zk, cfg)
        np.savetxt("./outputs/params_history_v", params_history_u)
        np.savetxt("./outputs/params_history_u", params_history_v)

        print("Learned params: ", params)
    
        return u_fn, v_fn
    
    def testGN(self, init_params, cfg):
        error = 1
        it = 0

        params = init_params
        uk_params, vk_params = params
        u_kernel = self.kernel_generator()
        v_kernel = self.kernel_generator()

        zk = jnp.zeros([4 * self.M_Omega, ])  # represent the values of operators at the samples

        params_history_u = [uk_params]
        params_history_v = [vk_params] 
        while  it < cfg.epoch: #error > cfg.tol and iter < cfg.epoch:
            # The optimization problem at each GN iteration is formulated as the following optimization problem
            # (Jz + y)^T\Theta^{-1}(Jz + y).
            # The minimization problem admits an explicit solution
            # z = -(J^T\Theta^{-1}J)^{-1}J^T\Theta^{-1}y

            # zk_current represents the values of operators of the functions at the previous iterations
            # evaluated at samples in the current iteration
            zk_current = zk
            zl_zeros = jnp.zeros([4*self.M_Omega, ])
            uk_params, vk_params = params

            y = self.prolong(zl_zeros, zk_current, self.M_Omega)
            J = jacfwd(self.prolong)(zl_zeros, zk_current, self.M_Omega)
            y1, y2 = jnp.split(y, 2, axis=0)
            J1, J2 = jnp.split(J, 2, axis=0)
            
            @jax.jit
            def solve(params):
                params_u, params_v = params
                theta_u = self.build_theta(
                    u_kernel, self.samples, params_u,
                )
                theta_v = self.build_theta(
                    v_kernel, self.samples, params_v,
                )
                nuggets_u = self.build_nuggets(theta_u)
                nuggets_v = self.build_nuggets(theta_v)
                gram_u = theta_u + cfg.nugget * nuggets_u
                gram_v = theta_v + cfg.nugget * nuggets_v
                cho_u = jax.scipy.linalg.cho_factor(gram_u)
                cho_v = jax.scipy.linalg.cho_factor(gram_v)

                A = jnp.dot(J1.T, jax.scipy.linalg.cho_solve(cho_u, J1)) + \
                    jnp.dot(J2.T, jax.scipy.linalg.cho_solve(cho_v, J2))
                b = -jnp.dot(J1.T, jax.scipy.linalg.cho_solve(cho_u, y1)) + \
                    -jnp.dot(J2.T, jax.scipy.linalg.cho_solve(cho_v, y2))
                new_z = jnp.linalg.solve(A, b)
                new_uvz = self.prolong(new_z, zk_current, self.M_Omega)
                new_uz, new_vz = jnp.split(new_uvz, 2, axis=0)

                u_weights = jax.scipy.linalg.cho_solve(cho_u, new_uz)
                v_weights = jax.scipy.linalg.cho_solve(cho_v, new_vz)
                res = self.nonlinear_residual(new_z, zk_current, self.M_Omega)
                # res = self.nonlinear_residual(new_z, self.M_Omega)
                error = jnp.dot(res, res)
                return error, u_weights, v_weights, new_z


            # res = self.nonlinear_residual(new_z, self.M_Omega)
            error, _, _, new_z = solve(params)

            it = it + 1

            print(f"GN Epoch {it}, param {params}, PDE Error {error}", flush=True)

            # making plots
            u_fn, v_fn = self.gen_uv(params, new_z, new_z, cfg)
            tt = jnp.linspace(0, 1, 101)
            xx = jnp.linspace(-5, 5, 129)[:-1]
            TT, XX = jnp.meshgrid(tt, xx)
            X_test = jnp.concatenate([TT.reshape([-1, 1]), XX.reshape([-1, 1])], axis=1)
            u_pred = vmap(u_fn)(X_test)
            v_pred = vmap(v_fn)(X_test)

            u_pred = u_pred.reshape([128, 101])
            v_pred = v_pred.reshape([128, 101])
            xx_test = XX.copy()
            tt_test = TT.copy()

            xx_test = np.array(xx_test)
            tt_test = np.array(tt_test)
            u_pred = np.array(u_pred)
            v_pred = np.array(v_pred)
            
            plt.figure()
            plt.pcolormesh(
                tt_test, 
                xx_test, 
                u_pred,
                cmap=colormaps["jet"],
            )
            plt.xlabel("$t$")
            plt.ylabel("$x$")
            plt.axis("equal")
            plt.title("$u$ (GP)")
            plt.savefig("./figures/u_gp.png")
            plt.close()

            plt.figure()
            plt.pcolormesh(
                tt_test, 
                xx_test, 
                v_pred,
                cmap=colormaps["jet"],
            )
            plt.xlabel("$t$")
            plt.ylabel("$x$")
            plt.axis("equal")
            plt.title("$v$ (GP)")
            plt.savefig("./figures/v_gp.png")
            plt.close()

            params = jax.lax.stop_gradient(params)
            zk = jax.lax.stop_gradient(new_z)


        u_fn, v_fn = self.gen_uv(params, zk, zk, cfg)
        np.savetxt("./outputs/params_history_v", params_history_u)
        np.savetxt("./outputs/params_history_u", params_history_v)

        print("Learned params: ", params)
    
        return u_fn, v_fn
