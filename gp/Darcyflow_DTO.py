import jax.numpy as jnp
from jax import grad, vmap, hessian, jit, jacfwd, value_and_grad
from functools import partial
import jax.ops as jop
from jax import config
import jax.scipy as jsp
config.update("jax_enable_x64", True)
# numpy
import numpy as np
from numpy import random
import optax
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

class Darcyflow(object):
    def __init__(self, kernel_u_generator, kernel_a_generator, domain):
        self.domain = domain
        self.kernel_u_generator = kernel_u_generator
        self.kernel_a_generator = kernel_a_generator

    def f(self, x1, x2):
        return 1

    # def f(self, x1, x2):
    #     return jnp.sin(2 * jnp.pi * x1) * jnp.cos(4 * jnp.pi * x2)

    def g(self, x1, x2):
        return 0

    def set_samples(self, M, M_Omega, N_data, samples):
        self.samples = samples
        self.M = M
        self.M_Omega = M_Omega
        self.X_data = samples[0:N_data, :]
        self.N_data = N_data  # the first N_data-th points are selected as the observed data points
        self.prepare_data()

    def sampling(self, cfg):
        self.samples = self.domain.sampling(cfg.M, cfg.M_Omega)
        self.M = cfg.M
        self.M_Omega = cfg.M_Omega
        self.X_data = self.samples[0:cfg.N_data, :]
        self.N_data = cfg.N_data    #the first N_data-th points are selected as the observed data points
        self.prepare_data()

        # sample all the validation points in the beginning
        self.total_valid_samples = self.domain.sampling(cfg.valid_M, cfg.valid_M_Omega)

    def prepare_data(self):
        self.fxomega = vmap(self.f)(self.samples[:self.M_Omega, 0], self.samples[:self.M_Omega, 1])
        self.gbdr = np.zeros(self.M - self.M_Omega)

    def get_observation(self, data_u, noise_level):
        self.data_u = data_u + noise_level * random.normal(0, 1.0, jnp.shape(data_u)[0])
        self.noise_level = noise_level


    def uKnm(self, kernel_u, xl, Ml, M_Omegal, xr, Mr, M_Omegar):
        xlint0 = jnp.reshape(xl[:M_Omegal, 0], (M_Omegal, 1))
        xlint1 = jnp.reshape(xl[:M_Omegal, 1], (M_Omegal, 1))
        xl0 = jnp.reshape(xl[:, 0], (Ml, 1))
        xl1 = jnp.reshape(xl[:, 1], (Ml, 1))

        xrint0 = jnp.reshape(xr[:M_Omegar, 0], (M_Omegar, 1))
        xrint1 = jnp.reshape(xr[:M_Omegar, 1], (M_Omegar, 1))
        xr0 = jnp.reshape(xr[:, 0], (Mr, 1))
        xr1 = jnp.reshape(xr[:, 1], (Mr, 1))

        xlxr0v = jnp.tile(xl0, Mr).flatten()
        xlxr0h = jnp.tile(np.transpose(xr0), (Ml, 1)).flatten()
        xlxr1v = jnp.tile(xl1, Mr).flatten()
        xlxr1h = jnp.tile(np.transpose(xr1), (Ml, 1)).flatten()

        xlxrint0v = jnp.tile(xl0, M_Omegar).flatten()
        xlxrint0h = jnp.tile(jnp.transpose(xrint0), (Ml, 1)).flatten()
        xlxrint1v = jnp.tile(xl1, M_Omegar).flatten()
        xlxrint1h = jnp.tile(jnp.transpose(xrint1), (Ml, 1)).flatten()

        xlintxr0v = jnp.tile(xlint0, Mr).flatten()
        xlintxr0h = jnp.tile(jnp.transpose(xr0), (M_Omegal, 1)).flatten()
        xlintxr1v = jnp.tile(xlint1, Mr).flatten()
        xlintxr1h = jnp.tile(jnp.transpose(xr1), (M_Omegal, 1)).flatten()

        xlintxrint0v = jnp.tile(xlint0, M_Omegar).flatten()
        xlintxrint0h = jnp.tile(jnp.transpose(xrint0), (M_Omegal, 1)).flatten()
        xlintxrint1v = jnp.tile(xlint1, M_Omegar).flatten()
        xlintxrint1h = jnp.tile(jnp.transpose(xrint1), (M_Omegal, 1)).flatten()

        Theta_u = jnp.zeros((3 * M_Omegal + Ml, 3 * M_Omegar + Mr))

        # Construct kernel matrix Theta_u
        # interior v.s. interior
        val = vmap(lambda x1, x2, y1, y2: kernel_u.D_x1_D_y1_kappa(x1, x2, y1, y2))(xlintxrint0v, xlintxrint1v,
                                                                                         xlintxrint0h, xlintxrint1h)
        Theta_u = Theta_u.at[0:M_Omegal, 0:M_Omegar].set(jnp.reshape(val, (M_Omegal, M_Omegar)))

        val = vmap(lambda x1, x2, y1, y2: kernel_u.D_x2_D_y2_kappa(x1, x2, y1, y2))(xlintxrint0v, xlintxrint1v,
                                                                                         xlintxrint0h, xlintxrint1h)
        Theta_u = Theta_u.at[M_Omegal:2 * M_Omegal, M_Omegar:2 * M_Omegar].set(jnp.reshape(val, (M_Omegal, M_Omegar)))

        val = vmap(lambda x1, x2, y1, y2: kernel_u.D_x2_D_y1_kappa(x1, x2, y1, y2))(xlintxrint0v, xlintxrint1v,
                                                                                         xlintxrint0h, xlintxrint1h)

        Theta_u = Theta_u.at[M_Omegal:2 * M_Omegal, 0:M_Omegar].set(jnp.reshape(val, (M_Omegal, M_Omegar)))

        val = vmap(lambda x1, x2, y1, y2: kernel_u.D_x1_D_y2_kappa(x1, x2, y1, y2))(xlintxrint0v, xlintxrint1v,
                                                                                         xlintxrint0h, xlintxrint1h)

        Theta_u = Theta_u.at[0:M_Omegal, M_Omegar:2 * M_Omegar].set(jnp.reshape(val, (M_Omegal, M_Omegar)))

        val = vmap(lambda x1, x2, y1, y2: kernel_u.Delta_x_Delta_y_kappa(x1, x2, y1, y2))(xlintxrint0v, xlintxrint1v,
                                                                                               xlintxrint0h, xlintxrint1h)
        Theta_u = Theta_u.at[2 * M_Omegal:3 * M_Omegal, 2 * M_Omegar:3 * M_Omegar].set(jnp.reshape(val, (M_Omegal, M_Omegar)))

        val = vmap(lambda x1, x2, y1, y2: kernel_u.Delta_x_D_y1_kappa(x1, x2, y1, y2))(xlintxrint0v, xlintxrint1v,
                                                                                            xlintxrint0h, xlintxrint1h)
        Theta_u = Theta_u.at[2 * M_Omegal:3 * M_Omegal, 0:M_Omegar].set(jnp.reshape(val, (M_Omegal, M_Omegar)))

        val = vmap(lambda x1, x2, y1, y2: kernel_u.D_x1_Delta_y_kappa(x1, x2, y1, y2))(xlintxrint0v, xlintxrint1v,
                                                                                            xlintxrint0h, xlintxrint1h)
        Theta_u = Theta_u.at[0:M_Omegal, 2 * M_Omegar:3 * M_Omegar].set((jnp.reshape(val, (M_Omegal, M_Omegar))))


        val = vmap(lambda x1, x2, y1, y2: kernel_u.Delta_x_D_y2_kappa(x1, x2, y1, y2))(xlintxrint0v, xlintxrint1v,
                                                                                            xlintxrint0h, xlintxrint1h)
        Theta_u = Theta_u.at[2 * M_Omegal:3 * M_Omegal, M_Omegar:2 * M_Omegar].set(jnp.reshape(val, (M_Omegal, M_Omegar)))

        val = vmap(lambda x1, x2, y1, y2: kernel_u.D_x2_Delta_y_kappa(x1, x2, y1, y2))(xlintxrint0v, xlintxrint1v,
                                                                                            xlintxrint0h, xlintxrint1h)

        Theta_u = Theta_u.at[M_Omegal:2 * M_Omegal, 2 * M_Omegar:3 * M_Omegar].set(jnp.reshape(val, (M_Omegal, M_Omegar)))

        # interior+boundary v.s. interior+boundary
        val = vmap(lambda x1, x2, y1, y2: kernel_u.kappa(x1, x2, y1, y2))(xlxr0v, xlxr1v, xlxr0h, xlxr1h)
        Theta_u = Theta_u.at[3 * M_Omegal:, 3 * M_Omegar:].set(jnp.reshape(val, (Ml, Mr)))

        # interior v.s. interior+boundary
        val = vmap(lambda x1, x2, y1, y2: kernel_u.D_x1_kappa(x1, x2, y1, y2))(xlintxr0v, xlintxr1v, xlintxr0h, xlintxr1h)
        Theta_u = Theta_u.at[0:M_Omegal, 3 * M_Omegar:].set(jnp.reshape(val, (M_Omegal, Mr)))

        val = vmap(lambda x1, x2, y1, y2: kernel_u.D_y1_kappa(x1, x2, y1, y2))(xlxrint0v, xlxrint1v, xlxrint0h, xlxrint1h)
        Theta_u = Theta_u.at[3 * M_Omegal:, 0:M_Omegar].set(jnp.reshape(val, (Ml, M_Omegar)))

        val = vmap(lambda x1, x2, y1, y2: kernel_u.D_x2_kappa(x1, x2, y1, y2))(xlintxr0v, xlintxr1v, xlintxr0h, xlintxr1h)
        Theta_u = Theta_u.at[M_Omegal:2 * M_Omegal, 3 * M_Omegar:].set(jnp.reshape(val, (M_Omegal, Mr)))

        val = vmap(lambda x1, x2, y1, y2: kernel_u.D_y2_kappa(x1, x2, y1, y2))(xlxrint0v, xlxrint1v, xlxrint0h, xlxrint1h)
        Theta_u = Theta_u.at[3 * M_Omegal:, M_Omegar:2 * M_Omegar].set(jnp.reshape(val, (Ml, M_Omegar)))

        val = vmap(lambda x1, x2, y1, y2: kernel_u.Delta_x_kappa(x1, x2, y1, y2))(xlintxr0v, xlintxr1v, xlintxr0h, xlintxr1h)
        Theta_u = Theta_u.at[2 * M_Omegal:3 * M_Omegal, 3 * M_Omegar:].set(jnp.reshape(val, (M_Omegal, Mr)))

        val = vmap(lambda x1, x2, y1, y2: kernel_u.Delta_y_kappa(x1, x2, y1, y2))(xlxrint0v, xlxrint1v, xlxrint0h, xlxrint1h)
        Theta_u = Theta_u.at[3 * M_Omegal:, 2 * M_Omegar:3 * M_Omegar].set(jnp.reshape(val, (Ml, M_Omegar)))
        return Theta_u

    def build_u_theta(self, u_kernel, x, M, M_Omega):
        return self.uKnm(u_kernel, x, M, M_Omega, x, M, M_Omega)

    def aKnm(self, kernel_a, xl, Ml, M_Omegal, xr, Mr, M_Omegar):
        xlint0 = jnp.reshape(xl[:M_Omegal, 0], (M_Omegal, 1))
        xlint1 = jnp.reshape(xl[:M_Omegal, 1], (M_Omegal, 1))
        xl0 = jnp.reshape(xl[:, 0], (Ml, 1))
        xl1 = jnp.reshape(xl[:, 1], (Ml, 1))

        xrint0 = jnp.reshape(xr[:M_Omegar, 0], (M_Omegar, 1))
        xrint1 = jnp.reshape(xr[:M_Omegar, 1], (M_Omegar, 1))
        xr0 = jnp.reshape(xr[:, 0], (Mr, 1))
        xr1 = jnp.reshape(xr[:, 1], (Mr, 1))

        xlxr0v = jnp.tile(xl0, Mr).flatten()
        xlxr0h = jnp.tile(np.transpose(xr0), (Ml, 1)).flatten()
        xlxr1v = jnp.tile(xl1, Mr).flatten()
        xlxr1h = jnp.tile(np.transpose(xr1), (Ml, 1)).flatten()

        xlxrint0v = jnp.tile(xl0, M_Omegar).flatten()
        xlxrint0h = jnp.tile(jnp.transpose(xrint0), (Ml, 1)).flatten()
        xlxrint1v = jnp.tile(xl1, M_Omegar).flatten()
        xlxrint1h = jnp.tile(jnp.transpose(xrint1), (Ml, 1)).flatten()

        xlintxr0v = jnp.tile(xlint0, Mr).flatten()
        xlintxr0h = jnp.tile(jnp.transpose(xr0), (M_Omegal, 1)).flatten()
        xlintxr1v = jnp.tile(xlint1, Mr).flatten()
        xlintxr1h = jnp.tile(jnp.transpose(xr1), (M_Omegal, 1)).flatten()

        xlintxrint0v = jnp.tile(xlint0, M_Omegar).flatten()
        xlintxrint0h = jnp.tile(jnp.transpose(xrint0), (M_Omegal, 1)).flatten()
        xlintxrint1v = jnp.tile(xlint1, M_Omegar).flatten()
        xlintxrint1h = jnp.tile(jnp.transpose(xrint1), (M_Omegal, 1)).flatten()

        Theta_a = jnp.zeros((3 * M_Omegal, 3 * M_Omegar))

        # Construct kernel matrix Theta_u
        # interior v.s. interior
        val = vmap(lambda x1, x2, y1, y2: kernel_a.D_x1_D_y1_kappa(x1, x2, y1, y2))(xlintxrint0v, xlintxrint1v,
                                                                                         xlintxrint0h, xlintxrint1h)
        Theta_a = Theta_a.at[0:M_Omegal, 0:M_Omegar].set(jnp.reshape(val, (M_Omegal, M_Omegar)))

        val = vmap(lambda x1, x2, y1, y2: kernel_a.D_x2_D_y2_kappa(x1, x2, y1, y2))(xlintxrint0v, xlintxrint1v,
                                                                                         xlintxrint0h, xlintxrint1h)
        Theta_a = Theta_a.at[M_Omegal:2 * M_Omegal, M_Omegar:2 * M_Omegar].set(jnp.reshape(val, (M_Omegal, M_Omegar)))

        val = vmap(lambda x1, x2, y1, y2: kernel_a.D_x2_D_y1_kappa(x1, x2, y1, y2))(xlintxrint0v, xlintxrint1v,
                                                                                         xlintxrint0h, xlintxrint1h)
        Theta_a = Theta_a.at[M_Omegal:2 * M_Omegal, 0:M_Omegar].set(jnp.reshape(val, (M_Omegal, M_Omegar)))

        val = vmap(lambda x1, x2, y1, y2: kernel_a.D_x1_D_y2_kappa(x1, x2, y1, y2))(xlintxrint0v, xlintxrint1v,
                                                                                         xlintxrint0h, xlintxrint1h)
        Theta_a = Theta_a.at[0:M_Omegal, M_Omegar:2 * M_Omegar].set(jnp.reshape(val, (M_Omegal, M_Omegar)))

        # interior+boundary v.s. interior+boundary
        val = vmap(lambda x1, x2, y1, y2: kernel_a.kappa(x1, x2, y1, y2))(xlintxrint0v, xlintxrint1v, xlintxrint0h,
                                                                               xlintxrint1h)
        Theta_a = Theta_a.at[2 * M_Omegal:, 2 * M_Omegar:].set(jnp.reshape(val, (M_Omegal, M_Omegar)))

        # interior v.s. interior+boundary
        val = vmap(lambda x1, x2, y1, y2: kernel_a.D_x1_kappa(x1, x2, y1, y2))(xlintxrint0v, xlintxrint1v, xlintxrint0h,
                                                                                    xlintxrint1h)
        Theta_a = Theta_a.at[0:M_Omegal, 2 * M_Omegar:].set(jnp.reshape(val, (M_Omegal, M_Omegar)))

        val = vmap(lambda x1, x2, y1, y2: kernel_a.D_y1_kappa(x1, x2, y1, y2))(xlintxrint0v, xlintxrint1v,
                                                                                    xlintxrint0h,
                                                                                    xlintxrint1h)
        Theta_a = Theta_a.at[2 * M_Omegal:, 0:M_Omegar].set(jnp.reshape(val, (M_Omegal, M_Omegar)))

        val = vmap(lambda x1, x2, y1, y2: kernel_a.D_x2_kappa(x1, x2, y1, y2))(xlintxrint0v, xlintxrint1v, xlintxrint0h,
                                                                                    xlintxrint1h)

        Theta_a = Theta_a.at[M_Omegal:2 * M_Omegal, 2 * M_Omegar:].set(jnp.reshape(val, (M_Omegal, M_Omegar)))

        val = vmap(lambda x1, x2, y1, y2: kernel_a.D_y2_kappa(x1, x2, y1, y2))(xlintxrint0v, xlintxrint1v,
                                                                                    xlintxrint0h,
                                                                                    xlintxrint1h)
        Theta_a = Theta_a.at[2 * M_Omegal:, M_Omegar:2 * M_Omegar].set(jnp.reshape(val, (M_Omegal, M_Omegar)))

        return Theta_a

    def build_a_theta(self, kernel_a, x, M, M_Omega):
        return self.aKnm(kernel_a, x, M, M_Omega, x, M, M_Omega)

    def build_u_nuggets(self, theta, M, M_Omega):
        trace1_u = jnp.trace(theta[:M_Omega, :M_Omega])
        trace2_u = jnp.trace(theta[M_Omega:2 * M_Omega, M_Omega:2 * M_Omega])
        trace3_u = jnp.trace(theta[2 * self.M_Omega:3 * self.M_Omega, 2 * self.M_Omega:3 * self.M_Omega])
        trace4_u = jnp.trace(theta[3 * self.M_Omega:, 3 * self.M_Omega:])
        ratio_u = [trace1_u / trace4_u, trace2_u / trace4_u, trace3_u / trace4_u]

        temp = jnp.concatenate((ratio_u[0] * jnp.ones((1, self.M_Omega)), ratio_u[1] * jnp.ones((1, self.M_Omega)), ratio_u[2] * jnp.ones((1, self.M_Omega)), jnp.ones((1, M))), axis=1)
        return jnp.diag(temp[0])

    def build_a_nuggets(self, theta, M, M_Omega):
        trace1_a = jnp.trace(theta[:M_Omega, :M_Omega])
        trace2_a = jnp.trace(theta[M_Omega:2 * M_Omega, M_Omega:2 * M_Omega])
        trace3_a = jnp.trace(theta[2 * M_Omega:3 * M_Omega, 2 * M_Omega:3 * M_Omega])
        ratio_a = [trace1_a / trace3_a, trace2_a / trace3_a]

        temp = jnp.concatenate((ratio_a[0] * jnp.ones((1, M_Omega)), ratio_a[1] * jnp.ones((1, M_Omega)), jnp.ones((1, M_Omega))), axis=1)
        return jnp.diag(temp[0])

    def build_gram(self, kernel_u, kernel_a, nugget=1e-8):
        theta_u = self.build_u_theta(kernel_u, self.samples, self.M, self.M_Omega)
        nuggets_u = self.build_u_nuggets(theta_u, self.M, self.M_Omega)
        self.Theta_u = theta_u + nugget * nuggets_u

        theta_a = self.build_a_theta(kernel_a, self.samples, self.M, self.M_Omega)
        nuggets_a = self.build_a_nuggets(theta_a, self.M, self.M_Omega)
        self.Theta_a = theta_a + nugget * nuggets_a


    def gram_Cholesky(self):
        self.L_u = jnp.linalg.cholesky(self.Theta_u)
        self.L_a = jnp.linalg.cholesky(self.Theta_a)

    def prolong(self, z, zk, M_Omega, fvals):
        # z contains the values of b and v, which are the solutions at the current GN iterations
        # zk contains the values of a and u, which are the solutions at the previous GN iterations
        # fvalues contains the values of f at the current iteration

        bx = z[0:M_Omega]  # bx
        by = z[M_Omega:2 * M_Omega]  # by
        b = z[2 * M_Omega:3 * M_Omega]  # b

        vx = z[3 * M_Omega:4 * M_Omega]  # vx
        vy = z[4 * M_Omega:5 * M_Omega]  # vy
        v = z[5 * self.M_Omega:] # v

        ax = zk[0:M_Omega]  # ax
        ay = zk[M_Omega:2 * M_Omega]  # ay
        a = zk[2 * M_Omega:3 * M_Omega]  # a

        ux = zk[3 * M_Omega:4 * M_Omega]  # ux
        uy = zk[4 * M_Omega:5 * M_Omega]  # uy
        u = zk[5 * M_Omega:] # u

        δa = b - a
        δax = bx - ax
        δay = by - ay
        δux = vx - ux
        δuy = vy - uy
        δv  = v - u

        # Δv = jnp.exp(-a) * δa * fvals - (ax * ux + ay * uy + jnp.exp(-a) * fvals + δax * ux + δay * uy + ax * δux + ay * δuy + u ** 3 + 3 * u **2 * δv)
        Δv = jnp.exp(-a) * δa * fvals - (ax * ux + ay * uy + jnp.exp(-a) * fvals + δax * ux + δay * uy + ax * δux + ay * δuy)
        return jnp.concatenate((bx, by, b, vx, vy, Δv, v, self.gbdr))

    def residual(self, z, zk, M_Omega, fvals):
        # z contains the values of b and v, which are the solutions at the current GN iterations
        # zk contains the values of a and u, which are the solutions at the previous GN iterations
        # fvalues contains the values of f at the current iteration

        bx = z[0:M_Omega]  # bx
        by = z[M_Omega:2 * M_Omega]  # by
        b = z[2 * M_Omega:3 * M_Omega]  # b

        vx = z[3 * M_Omega:4 * M_Omega]  # vx
        vy = z[4 * M_Omega:5 * M_Omega]  # vy
        Δv = z[5 * M_Omega:6 * M_Omega]  # Delta v
        v = z[6 * M_Omega:7 * M_Omega]

        ax = zk[0:M_Omega]  # ax
        ay = zk[M_Omega:2 * M_Omega]  # ay
        a = zk[2 * M_Omega:3 * M_Omega]  # a

        ux = zk[3 * M_Omega:4 * M_Omega]  # ux
        uy = zk[4 * M_Omega:5 * M_Omega]  # uy
        Δu = zk[5 * M_Omega:6 * M_Omega]  # Δu
        u = zk[6 * M_Omega:7 * M_Omega]  # u

        δa = b - a
        δax = bx - ax
        δay = by - ay
        δux = vx - ux
        δuy = vy - uy
        δv = v - u

        res = ax * ux + ay * uy + Δv + jnp.exp(-a) * fvals + δax * ux + δay * uy + ax * δux + ay * δuy - jnp.exp(-a) * δa * fvals #+ u ** 3 + 3 * u ** 2 * δv
        return res

    def nonlienar_residual(self, z, M_Omega, fvals):
        # z contains the values of b and v, which are the solutions at the current GN iterations
        # zk contains the values of a and u, which are the solutions at the previous GN iterations
        # fvalues contains the values of f at the current iteration

        b1 = z[0:M_Omega]  # bx
        b2 = z[M_Omega:2 * M_Omega]  # by
        b0 = z[2 * M_Omega:3 * M_Omega]  # b

        v1 = z[3 * M_Omega:4 * M_Omega]  # vx
        v2 = z[4 * M_Omega:5 * M_Omega]  # vy
        v3 = z[5 * M_Omega:6 * M_Omega]  # Delta v
        v = z[6 * M_Omega:7 * M_Omega]

        res = b1 * v1 + b2 * v2 + v3 + jnp.exp(-b0) * fvals #+ v**3
        return res

    def gen_a(self, params, z, cfg):
        kernel = self.kernel_a_generator(params)
        theta = self.build_a_theta(kernel, self.samples, self.M, self.M_Omega)
        nuggets = self.build_a_nuggets(theta, self.M, self.M_Omega)
        gram = theta + cfg.a_nugget * nuggets
        cho = jsp.linalg.cho_factor(gram)

        a1 = z[:self.M_Omega]  # ax
        a2 = z[self.M_Omega:2 * self.M_Omega]  # ay
        a0 = z[2 * self.M_Omega:3 * self.M_Omega]  # a
        az = jnp.concatenate((a1, a2, a0))

        a_weights = jsp.linalg.cho_solve(cho, az)
        return lambda x: jnp.dot(self.aKxPhi(kernel, x), a_weights)

    def gen_u(self, params, z, cfg):
        kernel = self.kernel_u_generator(params)
        theta = self.build_u_theta(kernel, self.samples, self.M, self.M_Omega)
        nuggets = self.build_u_nuggets(theta, self.M, self.M_Omega)
        gram = theta + cfg.u_nugget * nuggets
        cho = jsp.linalg.cho_factor(gram)

        u1 = z[3 * self.M_Omega:4 * self.M_Omega]  # ux
        u2 = z[4 * self.M_Omega:5 * self.M_Omega]  # uy
        u3 = z[5 * self.M_Omega:6 * self.M_Omega]  # Delta u
        u0 = z[6 * self.M_Omega:7 * self.M_Omega]  # u
        uz = jnp.concatenate((u1, u2, u3, u0, self.gbdr), axis=0)

        u_weights = jsp.linalg.cho_solve(cho, uz)
        return lambda x: jnp.dot(self.uKxPhi(kernel, x), u_weights)

    def uKxPhi(self, kernel_u, nx):
        x = self.samples
        xint0 = x[:self.M_Omega, 0]
        xint1 = x[:self.M_Omega, 1]
        x0 = x[:, 0]
        x1 = x[:, 1]

        ukxphi = jnp.zeros(3 * self.M_Omega + self.M)
        # constructing Theta_u_test matrix
        val = vmap(lambda y1, y2: kernel_u.D_y1_kappa(nx[0], nx[1], y1, y2))(xint0, xint1)
        ukxphi = ukxphi.at[:self.M_Omega].set(val)

        val = vmap(lambda y1, y2: kernel_u.D_y2_kappa(nx[0], nx[1], y1, y2))(xint0, xint1)
        ukxphi = ukxphi.at[self.M_Omega:2 * self.M_Omega].set(val)

        val = vmap(lambda y1, y2: kernel_u.Delta_y_kappa(nx[0], nx[1], y1, y2))(xint0, xint1)
        ukxphi = ukxphi.at[2 * self.M_Omega:3 * self.M_Omega].set(val)

        val = vmap(lambda y1, y2: kernel_u.kappa(nx[0], nx[1], y1, y2))(x0, x1)
        ukxphi = ukxphi.at[3 * self.M_Omega:].set(val)

        return ukxphi

    def aKxPhi(self, kernel_a, nx):
        x = self.samples
        xint0 = x[:self.M_Omega, 0]
        xint1 = x[:self.M_Omega, 1]

        aKxphi = jnp.zeros(3 * self.M_Omega)
        # constructing Theta_a_test matrix
        val = vmap(lambda y1, y2: kernel_a.D_y1_kappa(nx[0], nx[1], y1, y2))(xint0, xint1)
        aKxphi = aKxphi.at[:self.M_Omega].set(val)

        val = vmap(lambda y1, y2: kernel_a.D_y2_kappa(nx[0], nx[1], y1, y2))(xint0, xint1)
        aKxphi = aKxphi.at[self.M_Omega:2 * self.M_Omega].set(val)

        val = vmap(lambda y1, y2: kernel_a.kappa(nx[0], nx[1], y1, y2))(xint0, xint1)
        aKxphi = aKxphi.at[2 * self.M_Omega:].set(val)

        return aKxphi

    def trainGN(self, cfg):
        error = 1
        iter = 0

        uparams = cfg.ls_u
        aparams = cfg.ls_a

        uk_params  = uparams
        ak_params  = aparams
        uk_kernel = self.kernel_u_generator(uk_params)
        ak_kernel = self.kernel_a_generator(ak_params)
        uk_weights = jnp.zeros(3 * self.M_Omega + self.M)
        ak_weights = jnp.zeros(3 * self.M_Omega)

        params = jnp.array([cfg.ls_u, cfg.ls_a])

        zk = jnp.zeros(6 * self.M_Omega) # represent the values of operators at the samples
        while  iter < cfg.epoch: #error > cfg.tol and iter < cfg.epoch:
            f_vals = vmap(self.f)(self.samples[:self.M_Omega, 0], self.samples[:self.M_Omega, 1])
            # zk_current represents the values of operators of the functions at the previous iterations
            # evaluated at samples in the current iteration
            zk_current = zk
            zl_zeros = jnp.zeros(6 * self.M_Omega)

            y = self.prolong(zl_zeros, zk_current, self.M_Omega, f_vals)
            J = jacfwd(self.prolong)(zl_zeros, zk_current, self.M_Omega, f_vals)

            B = jnp.concatenate((jnp.zeros((self.N_data, 5 * self.M_Omega)), jnp.eye(self.N_data),
                                 jnp.zeros((self.N_data, self.M_Omega - self.N_data))), axis=1)

            Ja = J[:3 * self.M_Omega, :]
            Ju = J[3 * self.M_Omega:, :]
            ya = y[:3 * self.M_Omega]
            yu = y[3 * self.M_Omega:]

            idx = np.random.choice(self.total_valid_samples.shape[0], self.total_valid_samples.shape[0], replace=False,)[: cfg.batch_size]
            batch_samples_interior_valid = self.total_valid_samples[idx, :]

            @jit
            def solve_pde(params):
                # -------------------------------------------------------------------
                # Objective:
                #   Find z = [z_a; z_u] ∈ ℝ^{6M_Ω} that minimizes
                #
                #     Φ(z) = ½·(J z − y)^T · Θ⁻¹ · (J z − y)
                #          + ½·σ⁻² · ‖B z_u − D‖²₂
                #
                #   where
                #     • J = ∂(prolong)/∂z  (Jacobian of the forward map)
                #     • y = prolong(z)     (current forward evaluation)
                #     • Θ⁻¹ = diag(Θ_a⁻¹, Θ_u⁻¹)  (block-diagonal prior precision)
                #     • B selects the first N_data entries of z_u
                #     • D is the observed data vector
                #     • σ² = noise_level**2
                #
                # Normal equations (set ∇Φ = 0):
                #   (J^T Θ⁻¹ J + σ⁻² B^T B) z = J^T Θ⁻¹ y  +  σ⁻² B^T D
                #
                # Solver strategy:
                #   1. Factor Θ_a = L_a L_a^T,  Θ_u = L_u L_u^T  via Cholesky
                #   2. Compute “whitened” Jacobians:
                #        M_a = L_a⁻¹ J_a,    M_u = L_u⁻¹ J_u
                #      so  J^T Θ⁻¹ J = M_a^T M_a + M_u^T M_u
                #   3. Assemble data term:
                #        C = σ⁻² (B^T B),   rhs_data = σ⁻² (B^T D)
                #   4. Compute rhs_pde = J^T Θ⁻¹ y by
                #        tmp_a = L_a⁻T (L_a⁻¹ y_a),   tmp_u = L_u⁻T (L_u⁻¹ y_u)
                #        rhs_pde = J^T [tmp_a; tmp_u]
                #   5. Form
                #        A = (M_a^T M_a + M_u^T M_u) + C
                #        rhs = rhs_data − rhs_pde
                #   6. Solve A z = rhs via Cholesky (cho_factor + cho_solve)
                #
                # This yields the Gauss–Newton update z_new, which is then unrolled
                # into z_a and z_u for the next iteration.
                # -------------------------------------------------------------------
                u_params, a_params = params

                u_kernel = self.kernel_u_generator(u_params)
                a_kernel = self.kernel_a_generator(a_params)
                theta_u = self.build_u_theta(u_kernel, self.samples, self.M, self.M_Omega)
                nuggets_u = self.build_u_nuggets(theta_u, self.M, self.M_Omega)
                gram_u = theta_u + cfg.u_nugget * nuggets_u
                L_u = jnp.linalg.cholesky(gram_u)

                theta_a = self.build_a_theta(a_kernel, self.samples, self.M, self.M_Omega)
                nuggets_a = self.build_a_nuggets(theta_a, self.M, self.M_Omega)
                gram_a = theta_a + cfg.a_nugget * nuggets_a
                L_a = jnp.linalg.cholesky(gram_a)

                L_a_inv_Ja = jsp.linalg.solve_triangular(L_a, Ja, lower=True)
                L_u_inv_Ju = jsp.linalg.solve_triangular(L_u, Ju, lower=True)

                mtx1 = L_a_inv_Ja.T @ L_a_inv_Ja + L_u_inv_Ju.T @ L_u_inv_Ju
                mtx2 = 1 / self.noise_level ** 2 * jnp.matmul(B.T, B)

                rhs1 = 1 / self.noise_level ** 2 * B.T @ self.data_u
                tmp1 = jsp.linalg.solve_triangular(L_a.T, jsp.linalg.solve_triangular(L_a, ya, lower=True), lower=False)
                tmp2 = jsp.linalg.solve_triangular(L_u.T, jsp.linalg.solve_triangular(L_u, yu, lower=True), lower=False)
                rhs2 = J.T @ jnp.concatenate((tmp1, tmp2))
                rhs = rhs1 - rhs2
                mtx = mtx1 + mtx2
                cho = jsp.linalg.cho_factor(mtx)

                new_z = jsp.linalg.cho_solve(cho, rhs)

                new_full_z = self.prolong(new_z, zk_current, self.M_Omega, f_vals)
                new_az = new_full_z[:3 * self.M_Omega]
                new_uz = new_full_z[3 * self.M_Omega:]

                u_weights = jsp.linalg.solve_triangular(L_u.T, jsp.linalg.solve_triangular(L_u, new_uz, lower=True),
                                                        lower=False)
                a_weights = jsp.linalg.solve_triangular(L_a.T, jsp.linalg.solve_triangular(L_a, new_az, lower=True),
                                                        lower=False)
                return new_z[3 * self.M_Omega:6 * self.M_Omega], u_weights,  new_az, a_weights

            def loss_learning_params(params, valid_samples):
                M_Omega = cfg.batch_size + self.N_data

                u_params, a_params = params
                u_kernel = self.kernel_u_generator(u_params)
                a_kernel = self.kernel_a_generator(a_params)

                new_uz, u_weights,  new_az, a_weights = solve_pde(params)
                new_z = jnp.concatenate((new_az, new_uz))

                obs = new_z[5 * self.M_Omega:5 * self.M_Omega + self.N_data]

                u_valid_vals, dx1_u_valid_vals, dx2_u_valid_vals, Delta_u_valid_vals = self.eval_u_ops(u_kernel, u_weights, valid_samples)
                a_valid_vals, dx1_a_valid_vals, dx2_a_valid_vals = self.eval_a_ops(a_kernel, a_weights, valid_samples)
                z_at_valid = jnp.concatenate((dx1_a_valid_vals, dx2_a_valid_vals, a_valid_vals, dx1_u_valid_vals, dx2_u_valid_vals, Delta_u_valid_vals, u_valid_vals))

                uk_vals, dx1_uk_vals, dx2_uk_vals, Delta_uk_vals = self.eval_u_ops(uk_kernel, uk_weights, valid_samples)
                ak_vals, dx1_ak_vals, dx2_ak_vals = self.eval_a_ops(ak_kernel, ak_weights, valid_samples)
                zk_at_valid = jnp.concatenate((dx1_ak_vals, dx2_ak_vals, ak_vals, dx1_uk_vals, dx2_uk_vals, Delta_uk_vals, uk_vals))

                f_valid_vals = vmap(self.f)(valid_samples[:, 0], valid_samples[:, 1])
                res = self.residual(z_at_valid, zk_at_valid, M_Omega, f_valid_vals)
                error = jnp.dot(res, res) + 1/self.noise_level ** 2 * jnp.dot(obs - self.data_u, obs - self.data_u)
                return error

            # params = jnp.array([cfg.ls_u, cfg.ls_a])
            optimizer = optax.adam(learning_rate=1e-3)
            opt_state = optimizer.init(params)

            valid_samples = jnp.concatenate((self.samples[:self.N_data, :], batch_samples_interior_valid), axis=0)

            @jit
            def compute_updates(params, valid_samples, opt_state):
                current_loss, grads = value_and_grad(loss_learning_params)(params, valid_samples)
                # Update parameters and return new state
                updates, opt_state = optimizer.update(grads, opt_state, params)
                return updates, opt_state, current_loss

            for epoch in range(cfg.learning_epoch):
                # valid_samples = self.domain.sampling(cfg.valid_M, cfg.valid_M_Omega)
                # valid_samples = jnp.concatenate((self.samples[:self.N_data, :], valid_samples[:cfg.valid_M_Omega, :]),
                #                                 axis=0)
                valid_M = cfg.batch_size + self.N_data
                valid_M_Omega = cfg.batch_size + self.N_data

                # current_loss, grads = value_and_grad(loss_learning_params)(params, valid_samples, valid_M, valid_M_Omega)
                # # Update parameters and return new state
                # updates, opt_state = optimizer.update(grads, opt_state, params)

                updates, opt_state, current_loss = compute_updates(params, valid_samples, opt_state)

                params = optax.apply_updates(params, updates)
                print(
                    f"        Learning Hyperparameters, GN Epoch {iter}, Learning Hyper Epoch {epoch}, Current Loss {current_loss}, param u {params[0]}, param a {params[1]}")

                if current_loss <= cfg.learning_tol:
                    print(f"Stopping early at epoch {epoch} due to loss threshold")
                    break


            new_uz, u_weights, new_az, a_weights = solve_pde(params)
            new_z = jnp.concatenate((new_az, new_uz))

            res = self.nonlienar_residual(new_z, self.M_Omega, self.fxomega)
            error = jnp.dot(res, res)

            iter = iter + 1

            print(f"Epoch {iter}, param u {params[0]}, param a {params[1]}, PDE error {error}")

            u_params, a_params = params
            u_kernel = self.kernel_u_generator(u_params)
            a_kernel = self.kernel_a_generator(a_params)

            uk_params = params[0]
            ak_params = params[1]
            uk_kernel = u_kernel
            ak_kernel = a_kernel
            uk_weights = u_weights
            ak_weights = a_weights
            zk = new_z

        z_full = self.prolong(zk, zk, self.M_Omega, self.fxomega)
        u_func = self.gen_u(uk_params, z_full, cfg)
        a_func = self.gen_a(ak_params, z_full, cfg)
        return u_func, a_func


    def eval_u_ops(self, kernel_u, u_weights, nx):
        x = self.samples
        xint0 = np.reshape(x[:self.M_Omega, 0], (self.M_Omega, 1))
        xint1 = np.reshape(x[:self.M_Omega, 1], (self.M_Omega, 1))
        x0 = np.reshape(x[:, 0], (self.M, 1))
        x1 = np.reshape(x[:, 1], (self.M, 1))

        nxl = len(nx)
        nx0 = np.reshape(nx[:, 0], (nxl, 1))
        nx1 = np.reshape(nx[:, 1], (nxl, 1))

        xx0v = jnp.tile(nx0, self.M).flatten()
        xx0h = jnp.tile(np.transpose(x0), (nxl, 1)).flatten()
        xx1v = jnp.tile(nx1, self.M).flatten()
        xx1h = jnp.tile(np.transpose(x1), (nxl, 1)).flatten()

        xxint0v = jnp.tile(nx0, self.M_Omega).flatten()
        xxint0h = jnp.tile(jnp.transpose(xint0), (nxl, 1)).flatten()
        xxint1v = jnp.tile(nx1, self.M_Omega).flatten()
        xxint1h = jnp.tile(jnp.transpose(xint1), (nxl, 1)).flatten()

        # Compute u
        Theta_u_test = jnp.zeros((nxl, 3 * self.M_Omega + self.M))
        val = vmap(lambda x1, x2, y1, y2: kernel_u.D_y1_kappa(x1, x2, y1, y2))(xxint0v, xxint1v, xxint0h, xxint1h)
        Theta_u_test = Theta_u_test.at[:, :self.M_Omega].set(jnp.reshape(val, (nxl, self.M_Omega)))

        val = vmap(lambda x1, x2, y1, y2: kernel_u.D_y2_kappa(x1, x2, y1, y2))(xxint0v, xxint1v, xxint0h, xxint1h)
        Theta_u_test = Theta_u_test.at[:, self.M_Omega:2 * self.M_Omega].set(jnp.reshape(val, (nxl, self.M_Omega)))

        val = vmap(lambda x1, x2, y1, y2: kernel_u.Delta_y_kappa(x1, x2, y1, y2))(xxint0v, xxint1v, xxint0h, xxint1h)
        Theta_u_test = Theta_u_test.at[:, 2 * self.M_Omega:3 * self.M_Omega].set(jnp.reshape(val, (nxl, self.M_Omega)))

        val = vmap(lambda x1, x2, y1, y2: kernel_u.kappa(x1, x2, y1, y2))(xx0v, xx1v, xx0h, xx1h)
        Theta_u_test = Theta_u_test.at[:, 3 * self.M_Omega:].set(jnp.reshape(val, (nxl, self.M)))

        u_vals = jnp.matmul(Theta_u_test, u_weights)

        # Compute dx1_u
        Theta_u_test = jnp.zeros((nxl, 3 * self.M_Omega + self.M))
        val = vmap(lambda x1, x2, y1, y2: kernel_u.D_x1_D_y1_kappa(x1, x2, y1, y2))(xxint0v, xxint1v, xxint0h, xxint1h)
        Theta_u_test = Theta_u_test.at[:, :self.M_Omega].set(jnp.reshape(val, (nxl, self.M_Omega)))

        val = vmap(lambda x1, x2, y1, y2: kernel_u.D_x1_D_y2_kappa(x1, x2, y1, y2))(xxint0v, xxint1v, xxint0h, xxint1h)
        Theta_u_test = Theta_u_test.at[:, self.M_Omega:2 * self.M_Omega].set(jnp.reshape(val, (nxl, self.M_Omega)))

        val = vmap(lambda x1, x2, y1, y2: kernel_u.D_x1_Delta_y_kappa(x1, x2, y1, y2))(xxint0v, xxint1v, xxint0h, xxint1h)
        Theta_u_test = Theta_u_test.at[:, 2 * self.M_Omega:3 * self.M_Omega].set(jnp.reshape(val, (nxl, self.M_Omega)))

        val = vmap(lambda x1, x2, y1, y2: kernel_u.D_x1_kappa(x1, x2, y1, y2))(xx0v, xx1v, xx0h, xx1h)
        Theta_u_test = Theta_u_test.at[:, 3 * self.M_Omega:].set(jnp.reshape(val, (nxl, self.M)))

        dx1_u_vals = jnp.matmul(Theta_u_test, u_weights)

        # Compute dx2_u
        Theta_u_test = jnp.zeros((nxl, 3 * self.M_Omega + self.M))
        val = vmap(lambda x1, x2, y1, y2: kernel_u.D_x2_D_y1_kappa(x1, x2, y1, y2))(xxint0v, xxint1v, xxint0h, xxint1h)
        Theta_u_test = Theta_u_test.at[:, :self.M_Omega].set(jnp.reshape(val, (nxl, self.M_Omega)))

        val = vmap(lambda x1, x2, y1, y2: kernel_u.D_x2_D_y2_kappa(x1, x2, y1, y2))(xxint0v, xxint1v, xxint0h, xxint1h)
        Theta_u_test = Theta_u_test.at[:, self.M_Omega:2 * self.M_Omega].set(jnp.reshape(val, (nxl, self.M_Omega)))

        val = vmap(lambda x1, x2, y1, y2: kernel_u.D_x2_Delta_y_kappa(x1, x2, y1, y2))(xxint0v, xxint1v, xxint0h,
                                                                                       xxint1h)
        Theta_u_test = Theta_u_test.at[:, 2 * self.M_Omega:3 * self.M_Omega].set(jnp.reshape(val, (nxl, self.M_Omega)))

        val = vmap(lambda x1, x2, y1, y2: kernel_u.D_x2_kappa(x1, x2, y1, y2))(xx0v, xx1v, xx0h, xx1h)
        Theta_u_test = Theta_u_test.at[:, 3 * self.M_Omega:].set(jnp.reshape(val, (nxl, self.M)))

        dx2_u_vals = jnp.matmul(Theta_u_test, u_weights)

        # Compute Delta_u
        Theta_u_test = jnp.zeros((nxl, 3 * self.M_Omega + self.M))
        val = vmap(lambda x1, x2, y1, y2: kernel_u.Delta_x_D_y1_kappa(x1, x2, y1, y2))(xxint0v, xxint1v, xxint0h, xxint1h)
        Theta_u_test = Theta_u_test.at[:, :self.M_Omega].set(jnp.reshape(val, (nxl, self.M_Omega)))

        val = vmap(lambda x1, x2, y1, y2: kernel_u.Delta_x_D_y2_kappa(x1, x2, y1, y2))(xxint0v, xxint1v, xxint0h, xxint1h)
        Theta_u_test = Theta_u_test.at[:, self.M_Omega:2 * self.M_Omega].set(jnp.reshape(val, (nxl, self.M_Omega)))

        val = vmap(lambda x1, x2, y1, y2: kernel_u.Delta_x_Delta_y_kappa(x1, x2, y1, y2))(xxint0v, xxint1v, xxint0h,
                                                                                       xxint1h)
        Theta_u_test = Theta_u_test.at[:, 2 * self.M_Omega:3 * self.M_Omega].set(jnp.reshape(val, (nxl, self.M_Omega)))

        val = vmap(lambda x1, x2, y1, y2: kernel_u.Delta_x_kappa(x1, x2, y1, y2))(xx0v, xx1v, xx0h, xx1h)
        Theta_u_test = Theta_u_test.at[:, 3 * self.M_Omega:].set(jnp.reshape(val, (nxl, self.M)))

        Delta_u_vals = jnp.matmul(Theta_u_test, u_weights)

        return u_vals, dx1_u_vals, dx2_u_vals, Delta_u_vals

    def eval_a_ops(self, kernel_a, a_weights, nx):
        x = self.samples
        xint0 = np.reshape(x[:self.M_Omega, 0], (self.M_Omega, 1))
        xint1 = np.reshape(x[:self.M_Omega, 1], (self.M_Omega, 1))
        x0 = np.reshape(x[:, 0], (self.M, 1))
        x1 = np.reshape(x[:, 1], (self.M, 1))

        nxl = len(nx)
        nx0 = np.reshape(nx[:, 0], (nxl, 1))
        nx1 = np.reshape(nx[:, 1], (nxl, 1))

        xx0v = jnp.tile(nx0, self.M).flatten()
        xx0h = jnp.tile(np.transpose(x0), (nxl, 1)).flatten()
        xx1v = jnp.tile(nx1, self.M).flatten()
        xx1h = jnp.tile(np.transpose(x1), (nxl, 1)).flatten()

        xxint0v = jnp.tile(nx0, self.M_Omega).flatten()
        xxint0h = jnp.tile(jnp.transpose(xint0), (nxl, 1)).flatten()
        xxint1v = jnp.tile(nx1, self.M_Omega).flatten()
        xxint1h = jnp.tile(jnp.transpose(xint1), (nxl, 1)).flatten()

        # compute a
        Theta_a_test = jnp.zeros((nxl, 3 * self.M_Omega))
        val = vmap(lambda x1, x2, y1, y2: kernel_a.D_y1_kappa(x1, x2, y1, y2))(xxint0v, xxint1v, xxint0h, xxint1h)
        Theta_a_test = Theta_a_test.at[:, :self.M_Omega].set(jnp.reshape(val, (nxl, self.M_Omega)))

        val = vmap(lambda x1, x2, y1, y2: kernel_a.D_y2_kappa(x1, x2, y1, y2))(xxint0v, xxint1v, xxint0h, xxint1h)
        Theta_a_test = Theta_a_test.at[:, self.M_Omega:2 * self.M_Omega].set(jnp.reshape(val, (nxl, self.M_Omega)))

        val = vmap(lambda x1, x2, y1, y2: kernel_a.kappa(x1, x2, y1, y2))(xxint0v, xxint1v, xxint0h, xxint1h)
        Theta_a_test = Theta_a_test.at[:, 2 * self.M_Omega:].set(jnp.reshape(val, (nxl, self.M_Omega)))

        a_vals = jnp.matmul(Theta_a_test, a_weights)

        # compute dx1_a
        Theta_a_test = jnp.zeros((nxl, 3 * self.M_Omega))
        val = vmap(lambda x1, x2, y1, y2: kernel_a.D_x1_D_y1_kappa(x1, x2, y1, y2))(xxint0v, xxint1v, xxint0h, xxint1h)
        Theta_a_test = Theta_a_test.at[:, :self.M_Omega].set(jnp.reshape(val, (nxl, self.M_Omega)))

        val = vmap(lambda x1, x2, y1, y2: kernel_a.D_x1_D_y2_kappa(x1, x2, y1, y2))(xxint0v, xxint1v, xxint0h, xxint1h)
        Theta_a_test = Theta_a_test.at[:, self.M_Omega:2 * self.M_Omega].set(jnp.reshape(val, (nxl, self.M_Omega)))

        val = vmap(lambda x1, x2, y1, y2: kernel_a.D_x1_kappa(x1, x2, y1, y2))(xxint0v, xxint1v, xxint0h, xxint1h)
        Theta_a_test = Theta_a_test.at[:, 2 * self.M_Omega:].set(jnp.reshape(val, (nxl, self.M_Omega)))

        dx1_a_vals = jnp.matmul(Theta_a_test, a_weights)

        # compute dx2_a
        Theta_a_test = jnp.zeros((nxl, 3 * self.M_Omega))
        val = vmap(lambda x1, x2, y1, y2: kernel_a.D_x2_D_y1_kappa(x1, x2, y1, y2))(xxint0v, xxint1v, xxint0h, xxint1h)
        Theta_a_test = Theta_a_test.at[:, :self.M_Omega].set(jnp.reshape(val, (nxl, self.M_Omega)))

        val = vmap(lambda x1, x2, y1, y2: kernel_a.D_x2_D_y2_kappa(x1, x2, y1, y2))(xxint0v, xxint1v, xxint0h, xxint1h)
        Theta_a_test = Theta_a_test.at[:, self.M_Omega:2 * self.M_Omega].set(jnp.reshape(val, (nxl, self.M_Omega)))

        val = vmap(lambda x1, x2, y1, y2: kernel_a.D_x2_kappa(x1, x2, y1, y2))(xxint0v, xxint1v, xxint0h, xxint1h)
        Theta_a_test = Theta_a_test.at[:, 2 * self.M_Omega:].set(jnp.reshape(val, (nxl, self.M_Omega)))

        dx2_a_vals = jnp.matmul(Theta_a_test, a_weights)

        return a_vals, dx1_a_vals, dx2_a_vals


