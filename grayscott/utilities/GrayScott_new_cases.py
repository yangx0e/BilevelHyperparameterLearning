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


class GrayScott(object):
    # Gray-Scott system: 
    # equation 1: u_t = Du u_xx - u v^2 + F (1 - u)
    # equation 2: v_t = Dv v_xx + u v^2 - (F + k) v
    def __init__(self, kernel_generator, domain):
        self.kernel_generator = kernel_generator
        self.domain = domain
        self.Du = 1e-3
        self.Dv = 2e-3
        self.F = 4e-2
        self.k = 6e-2

    def sampling_testing(self, cfg):
        # generate samples
        self.samples = self.domain.sampling(cfg.M, cfg.M_Omega)
        samples_c = self.samples[:cfg.M_Omega, :]
        samples_0b = self.samples[cfg.M_Omega:, :]
        ind0 = samples_0b[:, 0] == 0
        indb = (samples_0b[:, 1] == 0) | (samples_0b[:, 1] == 1)
        samples_0 = samples_0b[ind0]
        samples_b = samples_0b[indb]
        self.samples = jnp.concatenate((samples_c, samples_0, samples_b), axis=0)
        self.M = cfg.M
        self.M_Omega = cfg.M_Omega
        self.M_0 = samples_0.shape[0]
        self.M_b = samples_b.shape[0]

        t, x = jnp.split(self.samples, 2, axis=1)
        self.f1 = jnp.zeros([self.M_Omega, 1])
        self.f2 = jnp.zeros([self.M_Omega, 1])
        # case a:
        # u0 = sin(7pi x+pi/2), v0 = -cos(2pi x)
        # case b:
        # u0 = -cos(4pi x), v0 = sin(5pi x + pi/2)
        # case c:
        # u0 = cos(8pi x), v0 = cos(5pi x)

        # self.u0 = jnp.sin(7*np.pi*x[self.M_Omega:self.M_Omega+self.M_0] + np.pi/2)
        # self.v0 = -jnp.cos(2*np.pi*x[self.M_Omega:self.M_Omega+self.M_0])

        # self.u0 = -jnp.cos(4*np.pi*x[self.M_Omega:self.M_Omega+self.M_0])
        # self.v0 = jnp.sin(5*np.pi*x[self.M_Omega:self.M_Omega+self.M_0] + np.pi/2)

        self.u0 = jnp.cos(8*np.pi*x[self.M_Omega:self.M_Omega+self.M_0])
        self.v0 = jnp.cos(5*np.pi*x[self.M_Omega:self.M_Omega+self.M_0])
        
        
        self.u0 = self.u0.flatten()
        self.v0 = self.v0.flatten()
        self.dub = jnp.zeros([self.M_b])
        self.dvb = jnp.zeros([self.M_b])

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

        uxx = 1/self.Du * (ut + (vk**2 + self.F) * u + 2*uk*vk*v - 2*uk*vk**2 - self.F)
        vxx = 1/self.Dv * (-vk**2*u + vt + (-2*uk*vk + self.F + self.k)*v + 2*uk*vk**2)

        eq1 = ut - self.Du * uxx + u * v**2 - self.F * (1 - u)
        eq2 = vt - self.Dv * vxx - u * v**2 + (self.F + self.k) * v
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

        # ukt = zku[0:M_Omega]
        # ukxx = zku[M_Omega:2*M_Omega]
        uk = zku[2*M_Omega:3*M_Omega]
        # vkt = zkv[0:M_Omega]
        # vkxx = zkv[M_Omega:2*M_Omega]
        vk = zkv[2*M_Omega:3*M_Omega]

        eq1 = ut - self.Du * uxx + (vk**2 + self.F) * u + 2*uk*vk*v - 2*uk*vk**2 - self.F
        eq2 = -vk**2 * u + vt - self.Dv * vxx + (-2*uk*vk + self.F + self.k) * v + 2*uk*vk**2
        return jnp.concatenate([eq1, eq2], axis=0)
    
    def prolong(self, z, zk, M_Omega):
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

        uxx = 1/self.Du * (ut + (vk**2 + self.F) * u + 2*uk*vk*v - 2*uk*vk**2 - self.F)
        vxx = 1/self.Dv * (-vk**2*u + vt + (-2*uk*vk + self.F + self.k)*v + 2*uk*vk**2)
        rhs_1 = jnp.concatenate(
            [ut, uxx, u, self.u0, self.dub], axis=0,
        )
        rhs_2 = jnp.concatenate(
            [vt, vxx, v, self.v0, self.dvb], axis=0,
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
        M_b = self.M_b
        kxphi = jnp.zeros([1, 3*M_Omega+M_0+M_b])
        x1_c = x[:M_Omega, 0:1]
        x2_c = x[:M_Omega, 1:2]
        x1_0 = x[M_Omega:M_Omega+M_0, 0:1]
        x2_0 = x[M_Omega:M_Omega+M_0, 1:2]
        x1_b = x[M_Omega+M_0:, 0:1]
        x2_b = x[M_Omega+M_0:, 1:2]
        
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

        # u v.s. b
        x1_vb_l = jnp.tile(x1_v, [1, M_b]).flatten()
        x2_vb_l = jnp.tile(x2_v, [1, M_b]).flatten()
        x1_vb_r = x1_b.T.flatten()
        x2_vb_r = x2_b.T.flatten()

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
        # u vs ub
        val = vmap(
            lambda x1, x2, y1, y2: kernel.D_y2_kappa(x1, x2, y1, y2, params)
        )(x1_vb_l, x2_vb_l, x1_vb_r, x2_vb_r).reshape([1, M_b])
        kxphi = kxphi.at[:, 3*M_Omega+M_0:3*M_Omega+M_0+M_b].set(val)
        return kxphi.flatten()
    
    def build_theta(self, kernel, x, params):
        M_Omega = self.M_Omega
        M_0 = self.M_0
        M_b = self.M_b
        theta = jnp.zeros([3*M_Omega+M_0+M_b, 3*M_Omega+M_0+M_b])
        x1_c = x[:M_Omega, 0:1]
        x2_c = x[:M_Omega, 1:2]
        x1_0 = x[M_Omega:M_Omega+M_0, 0:1]
        x2_0 = x[M_Omega:M_Omega+M_0, 1:2]
        x1_b = x[M_Omega+M_0:, 0:1]
        x2_b = x[M_Omega+M_0:, 1:2]
        
        # ut, uxx, u, u0, dub
        # vt, vxx, v, v0, dvb
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

        # int v.s. b
        x1_cb_l = jnp.tile(x1_c, [1, M_b]).flatten()
        x2_cb_l = jnp.tile(x2_c, [1, M_b]).flatten()
        x1_cb_r = jnp.tile(x1_b.T, [M_Omega, 1]).flatten()
        x2_cb_r = jnp.tile(x2_b.T, [M_Omega, 1]).flatten()

        # 0 v.s. 0
        x1_00_l = jnp.tile(x1_0, [1, M_0]).flatten()
        x2_00_l = jnp.tile(x2_0, [1, M_0]).flatten()
        x1_00_r = jnp.tile(x1_0.T, [M_0, 1]).flatten()
        x2_00_r = jnp.tile(x2_0.T, [M_0, 1]).flatten()

        # 0 v.s. b
        x1_0b_l = jnp.tile(x1_0, [1, M_b]).flatten()
        x2_0b_l = jnp.tile(x2_0, [1, M_b]).flatten()
        x1_0b_r = jnp.tile(x1_b.T, [M_0, 1]).flatten()
        x2_0b_r = jnp.tile(x2_b.T, [M_0, 1]).flatten()

        # b v.s. b
        x1_bb_l = jnp.tile(x1_b, [1, M_b]).flatten()
        x2_bb_l = jnp.tile(x2_b, [1, M_b]).flatten()
        x1_bb_r = jnp.tile(x1_b.T, [M_b, 1]).flatten()
        x2_bb_r = jnp.tile(x2_b.T, [M_b, 1]).flatten()


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

        # ut vs ub
        val = vmap(
            lambda x1, x2, y1, y2: kernel.D_x1_D_y2_kappa(x1, x2, y1, y2, params)
        )(x1_cb_l, x2_cb_l, x1_cb_r, x2_cb_r).reshape([M_Omega, M_b])
        theta = theta.at[0:M_Omega, 3*M_Omega+M_0:3*M_Omega+M_0+M_b].set(val)
        theta = theta.at[3*M_Omega+M_0:3*M_Omega+M_0+M_b, 0:M_Omega].set(val.T)

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

        # uxx vs dub
        val = vmap(
            lambda x1, x2, y1, y2: kernel.DD_x2_D_y2_kappa(x1, x2, y1, y2, params)
        )(x1_cb_l, x2_cb_l, x1_cb_r, x2_cb_r).reshape([M_Omega, M_b])
        theta = theta.at[M_Omega:2*M_Omega, 3*M_Omega+M_0:3*M_Omega+M_0+M_b].set(val)
        theta = theta.at[3*M_Omega+M_0:3*M_Omega+M_0+M_b, M_Omega:2*M_Omega].set(val.T)

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

        # u vs dub
        val = vmap(
            lambda x1, x2, y1, y2: kernel.D_y2_kappa(x1, x2, y1, y2, params)
        )(x1_cb_l, x2_cb_l, x1_cb_r, x2_cb_r).reshape([M_Omega, M_b])
        theta = theta.at[2*M_Omega:3*M_Omega, 3*M_Omega+M_0:3*M_Omega+M_0+M_b].set(val)
        theta = theta.at[3*M_Omega+M_0:3*M_Omega+M_0+M_b, 2*M_Omega:3*M_Omega].set(val.T)

        #### part 4: u0 v.s. the rest
        # u0 vs u0
        val = vmap(
            lambda x1, x2, y1, y2: kernel.kappa(x1, x2, y1, y2, params)
        )(x1_00_l, x2_00_l, x1_00_r, x2_00_r).reshape([M_0, M_0])
        theta = theta.at[3*M_Omega:3*M_Omega+M_0, 3*M_Omega:3*M_Omega+M_0].set(val)

        # u0 vs dub
        val = vmap(
            lambda x1, x2, y1, y2: kernel.D_y2_kappa(x1, x2, y1, y2, params)
        )(x1_0b_l, x2_0b_l, x1_0b_r, x2_0b_r).reshape([M_0, M_b])
        theta = theta.at[3*M_Omega:3*M_Omega+M_0, 3*M_Omega+M_0:3*M_Omega+M_0+M_b].set(val)
        theta = theta.at[3*M_Omega+M_0:3*M_Omega+M_0+M_b, 3*M_Omega:3*M_Omega+M_0].set(val.T)

        #### part 5: dub v.s. the rest
        # dub vs dub
        val = vmap(
            lambda x1, x2, y1, y2: kernel.D_x2_D_y2_kappa(x1, x2, y1, y2, params)
        )(x1_bb_l, x2_bb_l, x1_bb_r, x2_bb_r).reshape([M_b, M_b])
        theta = theta.at[3*M_Omega+M_0:3*M_Omega+M_0+M_b, 3*M_Omega+M_0:3*M_Omega+M_0+M_b].set(val)
        return theta
    
    def build_nuggets(self, theta):
        M_Omega = self.M_Omega
        M_0 = self.M_0
        M_b = self.M_b
        trace11 = jnp.trace(theta[0:M_Omega, 0:M_Omega]) # ut
        trace22 = jnp.trace(theta[M_Omega:2*M_Omega, M_Omega:2*M_Omega]) # uxx
        trace33 = jnp.trace(theta[2*M_Omega:3*M_Omega, 2*M_Omega:3*M_Omega]) # u
        trace44 = jnp.trace(theta[3*M_Omega:3*M_Omega+M_0, 3*M_Omega:3*M_Omega+M_0]) # u0
        trace55 = jnp.trace(theta[3*M_Omega+M_0:3*M_Omega+M_0+M_b, 3*M_Omega+M_0:3*M_Omega+M_0+M_b]) # ub

        ratio = [
            trace11/(trace33+trace44), trace22/(trace33+trace44), trace55/(trace33+trace44),
        ]

        r_diag = jnp.concatenate(
            [
                ratio[0] * jnp.ones([1, M_Omega]),
                ratio[1] * jnp.ones([1, M_Omega]),
                jnp.ones([1, M_Omega]),
                jnp.ones([1, M_0]),
                ratio[2] * jnp.ones([1, M_b]),
            ], 
            axis=1,
        )
        r = jnp.diag(r_diag[0])
        return r
    
    def eval_ops(self, kernel, weights, nx, params):
        x = self.samples
        M_Omega = self.M_Omega
        M_0 = self.M_0
        M_b = self.M_b
        x1_c = x[:M_Omega, 0:1]
        x2_c = x[:M_Omega, 1:2]
        x1_0 = x[M_Omega:M_Omega+M_0, 0:1]
        x2_0 = x[M_Omega:M_Omega+M_0, 1:2]
        x1_b = x[M_Omega+M_0:, 0:1]
        x2_b = x[M_Omega+M_0:, 1:2]

        M_v = nx.shape[0]
        x1_v = nx[:, 0:1] # validation data
        x2_v = nx[:, 1:2] # validation data
        
        # ut, uxx, u, u0, dub
        # vt, vxx, v, v0, dvb
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

        # int v.s. b
        x1_vb_l = jnp.tile(x1_v, [1, M_b]).flatten()
        x2_vb_l = jnp.tile(x2_v, [1, M_b]).flatten()
        x1_vb_r = jnp.tile(x1_b.T, [M_v, 1]).flatten()
        x2_vb_r = jnp.tile(x2_b.T, [M_v, 1]).flatten()

        # ut vs ut, uxx, u, u0, dub
        mtx = jnp.zeros([M_v, 3*M_Omega+M_0+M_b])
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

        val = vmap(
            lambda x1, x2, y1, y2: kernel.D_x1_D_y2_kappa(x1, x2, y1, y2, params)
        )(x1_vb_l, x2_vb_l, x1_vb_r, x2_vb_r).reshape([M_v, M_b])
        mtx = mtx.at[:, 3*M_Omega+M_0:3*M_Omega+M_0+M_b].set(val)

        wt_v = mtx.dot(weights)

        # uxx vs ut, uxx, u, u0, dub
        mtx = jnp.zeros([M_v, 3*M_Omega+M_0+M_b])
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

        val = vmap(
            lambda x1, x2, y1, y2: kernel.DD_x2_D_y2_kappa(x1, x2, y1, y2, params)
        )(x1_vb_l, x2_vb_l, x1_vb_r, x2_vb_r).reshape([M_v, M_b])
        mtx = mtx.at[:, 3*M_Omega+M_0:3*M_Omega+M_0+M_b].set(val)

        wxx_v = mtx.dot(weights)

        # u vs ut, uxx, u, u0, dub
        mtx = jnp.zeros([M_v, 3*M_Omega+M_0+M_b])
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

        val = vmap(
            lambda x1, x2, y1, y2: kernel.D_y2_kappa(x1, x2, y1, y2, params)
        )(x1_vb_l, x2_vb_l, x1_vb_r, x2_vb_r).reshape([M_v, M_b])
        mtx = mtx.at[:, 3*M_Omega+M_0:3*M_Omega+M_0+M_b].set(val)

        w_v = mtx.dot(weights)

        ws = [wt_v, wxx_v, w_v]

        return ws
    
    def trainGN(self, init_params, cfg, params_reg_func = None):
        error = 1
        it = 0

        params = init_params
        uk_params, vk_params = params
        u_kernel = self.kernel_generator()
        v_kernel = self.kernel_generator()

        uk_weights = jnp.zeros(3*self.M_Omega + self.M_0 + self.M_b)
        vk_weights = jnp.zeros(3*self.M_Omega + self.M_0 + self.M_b)

        zk = jnp.zeros([4 * self.M_Omega, ])  # represent the values of operators at the samples
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

                
            error, u_weights, v_weights, new_z = solve(params)

            it = it + 1

            print(f"GN Epoch {it}, param {params}, PDE Error {error}", flush=True)

            params = jax.lax.stop_gradient(params)
            uk_weights = jax.lax.stop_gradient(u_weights)
            vk_weights = jax.lax.stop_gradient(v_weights)
            zk = jax.lax.stop_gradient(new_z)

        u_fn, v_fn = self.gen_uv(params, zk, zk, cfg)

        print("Learned params: ", params)
    
        return u_fn, v_fn
    
    def testGN(self, init_params, cfg, params_reg_func = None):
        error = 1
        it = 0

        params = init_params
        u_kernel = self.kernel_generator()
        v_kernel = self.kernel_generator()

        zk = jnp.zeros([4 * self.M_Omega, ])  # represent the values of operators at the samples
        while  it < cfg.epoch: #error > cfg.tol and iter < cfg.epoch:
            # The optimization problem at each GN iteration is formulated as the following optimization problem
            # (Jz + y)^T\Theta^{-1}(Jz + y).
            # The minimization problem admits an explicit solution
            # z = -(J^T\Theta^{-1}J)^{-1}J^T\Theta^{-1}y

            # zk_current represents the values of operators of the functions at the previous iterations
            # evaluated at samples in the current iteration
            zk_current = zk
            zl_zeros = jnp.zeros([4*self.M_Omega, ])

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

                
            error, _, _, new_z = solve(params)

            it = it + 1

            print(f"GN Epoch {it}, param {params}, PDE Error {error}", flush=True)

            params = jax.lax.stop_gradient(params)
            zk = jax.lax.stop_gradient(new_z)

        u_fn, v_fn = self.gen_uv(params, zk, zk, cfg)

        print("Learned params: ", params)
    
        return u_fn, v_fn
