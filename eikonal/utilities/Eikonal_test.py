import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import vmap, jacfwd, value_and_grad
from utilities.domains import *
import jax.scipy as jsp
import optax
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import scipy.io as sio


class Eikonal(object):
    # Eikonal equation: u_x^2 + u_y^2 = f + eps Delta u

    def __init__(self, kernel_generator, domain, ls_net):
        self.kernel_generator = kernel_generator
        self.domain = domain
        self.ls_net = ls_net

        self.f = 1
        self.eps = 0.01

    def sampling(self, cfg):
        self.samples = self.domain.sampling(cfg.M, cfg.M_Omega)
        self.M = cfg.M
        self.M_Omega = cfg.M_Omega
        self.samples = jnp.array(self.samples)
        self.gbdr = jnp.zeros([cfg.M - cfg.M_Omega])

        samples_interior_valid = self.domain.sampling(cfg.valid_M, cfg.valid_M_Omega)[:cfg.valid_M_Omega, :]

        samples_interior = self.samples[:cfg.M_Omega, :]
        samples_bdry = self.samples[cfg.M_Omega:, :]
        self.samples = jnp.concatenate(
            [samples_interior, samples_interior_valid, samples_bdry], axis=0,
        )
        self.M = cfg.M + cfg.valid_M_Omega
        self.M_Omega = cfg.M_Omega + cfg.valid_M_Omega

        sio.savemat(
            "./data/samples.mat",
            {
                "interior": np.array(self.samples[:self.M_Omega]),
                "boundary": np.array(self.samples[self.M_Omega:]),
            }
        )

    def residual(self, z, zk, M_Omega):
        # ux, uy, Delta u
        
        u_x = z[0:M_Omega]
        u_y = z[M_Omega:2*M_Omega]
        u_delta = z[2*M_Omega:3*M_Omega]

        uk_x = zk[0:M_Omega]
        uk_y = zk[M_Omega:2*M_Omega]
        # uk_delta = zk[2*M_Omega:3*M_Omega]

        out = 2 * uk_x * u_x - uk_x**2 + 2 * uk_y * u_y - uk_y**2 - self.f - self.eps * u_delta
        return out

    @partial(jax.jit, static_argnums=(0, 3))
    def prolong(self, z, zk, M_Omega):
        u_x = z[0:M_Omega]
        u_y = z[M_Omega:2*M_Omega]

        uk_x = zk[0:M_Omega]
        uk_y = zk[M_Omega:2*M_Omega]

        u_delta = (2 * uk_x * u_x - uk_x**2 + 2 * uk_y * u_y - uk_y**2 - self.f) / self.eps
        return jnp.concatenate([u_x, u_y, u_delta, self.gbdr])

    @partial(jax.jit, static_argnums=(0, 3))
    def nonlinear_residual(self, z, zk, M_Omega):
        u_x = z[0:M_Omega]
        u_y = z[M_Omega:2*M_Omega]

        uk_x = zk[0:M_Omega]
        uk_y = zk[M_Omega:2*M_Omega]
        u_delta = (2 * uk_x * u_x - uk_x**2 + 2 * uk_y * u_y - uk_y**2 - self.f) / self.eps

        out = u_x**2 + u_y**2 - self.f - self.eps * u_delta
        return out
    
    # @partial(jax.jit, static_argnums=(0))
    def gen_u(self, params, z, zk, nugget):
        kernel = self.kernel_generator()
        theta = self.build_theta(kernel, self.samples, params)
        nuggets = self.build_nuggets(theta, self.M, self.M_Omega)
        gram = theta + nugget * nuggets
        cho = jsp.linalg.cho_factor(gram)

        uz = self.prolong(z, zk, self.M_Omega)

        u_weights = jsp.linalg.cho_solve(cho, uz)
        return lambda x: jnp.dot(self.KxPhi(kernel, x, params), u_weights)

    def KxPhi(self, kernel, nx, params):
        x = self.samples
        xint0 = x[:self.M_Omega, 0]
        xint1 = x[:self.M_Omega, 1]
        xb0 = x[self.M_Omega:, 0]
        xb1 = x[self.M_Omega:, 1]

        kxphi = jnp.zeros(3*self.M_Omega+self.M-self.M_Omega)

        val = vmap(lambda y1, y2: kernel.D_y1_kappa(nx[0], nx[1], y1, y2, params))(xint0, xint1)
        kxphi = kxphi.at[0:self.M_Omega].set(val)

        val = vmap(lambda y1, y2: kernel.D_y2_kappa(nx[0], nx[1], y1, y2, params))(xint0, xint1)
        kxphi = kxphi.at[self.M_Omega:2*self.M_Omega].set(val)

        val = vmap(lambda y1, y2: kernel.Delta_y_kappa(nx[0], nx[1], y1, y2, params))(xint0, xint1)
        kxphi = kxphi.at[2*self.M_Omega:3*self.M_Omega].set(val)

        val = vmap(lambda y1, y2: kernel.kappa(nx[0], nx[1], y1, y2, params))(xb0, xb1)
        kxphi = kxphi.at[3*self.M_Omega:].set(val)

        return kxphi

    @partial(jax.jit, static_argnums=(0, 1))
    def build_theta(self, kernel, x, params):
        # u_x, u_y, delta_u, u
        M = self.M
        M_Omega = self.M_Omega
        theta = jnp.zeros([3*M_Omega+M-M_Omega, 3*M_Omega+M-M_Omega])

        xint0 = jnp.reshape(x[:M_Omega, 0], [M_Omega, 1])
        xint1 = jnp.reshape(x[:M_Omega, 1], [M_Omega, 1])
        xb0 = jnp.reshape(x[M_Omega:, 0], [M-M_Omega, 1])
        xb1 = jnp.reshape(x[M_Omega:, 1], [M-M_Omega, 1])

        # bdr v.s. bdr
        xxb0v = jnp.tile(xb0, M-M_Omega).flatten()
        xxb0h = jnp.tile(jnp.transpose(xb0), [M-M_Omega, 1]).flatten()
        xxb1v = jnp.tile(xb1, M-M_Omega).flatten()
        xxb1h = jnp.tile(jnp.transpose(xb1), [M-M_Omega, 1]).flatten()

        # int v.s. bdr
        xintxb0v = jnp.tile(xint0, M-M_Omega).flatten()
        xintxb0h = jnp.tile(jnp.transpose(xb0), [M_Omega, 1]).flatten()
        xintxb1v = jnp.tile(xint1, M-M_Omega).flatten()
        xintxb1h = jnp.tile(jnp.transpose(xb1), [M_Omega, 1]).flatten()

        # int v.s. int
        xintxint0v = jnp.tile(xint0, M_Omega).flatten()
        xintxint0h = jnp.tile(jnp.transpose(xint0), [M_Omega, 1]).flatten()
        xintxint1v = jnp.tile(xint1, M_Omega).flatten()
        xintxint1h = jnp.tile(jnp.transpose(xint1), [M_Omega, 1]).flatten()

        # u_x, u_y, Delta u, u
        val = vmap(lambda x1, x2, y1, y2: kernel.D_x1_D_y1_kappa(x1, x2, y1, y2, params))(xintxint0v, xintxint1v, xintxint0h, xintxint1h)
        theta = theta.at[0:M_Omega, 0:M_Omega].set(jnp.reshape(val, [M_Omega, M_Omega]))

        val = vmap(lambda x1, x2, y1, y2: kernel.D_x1_D_y2_kappa(x1, x2, y1, y2, params))(xintxint0v, xintxint1v, xintxint0h, xintxint1h)
        tmp = np.reshape(val, [M_Omega, M_Omega])
        theta = theta.at[0:M_Omega, M_Omega:2*M_Omega].set(tmp)
        theta = theta.at[M_Omega:2*M_Omega, 0:M_Omega].set(tmp.T)

        val = vmap(lambda x1, x2, y1, y2: kernel.D_x1_Delta_y_kappa(x1, x2, y1, y2, params))(xintxint0v, xintxint1v, xintxint0h, xintxint1h)
        tmp = np.reshape(val, [M_Omega, M_Omega])
        theta = theta.at[0:M_Omega, 2*M_Omega:3*M_Omega].set(tmp)
        theta = theta.at[2*M_Omega:3*M_Omega, 0:M_Omega].set(tmp.T)

        val = vmap(lambda x1, x2, y1, y2: kernel.D_x2_D_y2_kappa(x1, x2, y1, y2, params))(xintxint0v, xintxint1v, xintxint0h, xintxint1h)
        theta = theta.at[M_Omega:2*M_Omega, M_Omega:2*M_Omega].set(jnp.reshape(val, [M_Omega, M_Omega]))

        val = vmap(lambda x1, x2, y1, y2: kernel.D_x2_Delta_y_kappa(x1, x2, y1, y2, params))(xintxint0v, xintxint1v, xintxint0h, xintxint1h)
        tmp = np.reshape(val, [M_Omega, M_Omega])
        theta = theta.at[M_Omega:2*M_Omega, 2*M_Omega:3*M_Omega].set(tmp)
        theta = theta.at[2*M_Omega:3*M_Omega, M_Omega:2*M_Omega].set(tmp.T)

        val = vmap(lambda x1, x2, y1, y2: kernel.Delta_x_Delta_y_kappa(x1, x2, y1, y2, params))(xintxint0v, xintxint1v, xintxint0h, xintxint1h)
        theta = theta.at[2*M_Omega:3*M_Omega, 2*M_Omega:3*M_Omega].set(jnp.reshape(val, [M_Omega, M_Omega]))

        val = vmap(lambda x1, x2, y1, y2: kernel.kappa(x1, x2, y1, y2, params))(xxb0v, xxb1v, xxb0h, xxb1h)
        theta = theta.at[3*M_Omega:, 3*M_Omega:].set(jnp.reshape(val, [M-M_Omega, M-M_Omega]))

        val = vmap(lambda x1, x2, y1, y2: kernel.D_x1_kappa(x1, x2, y1, y2, params))(xintxb0v, xintxb1v, xintxb0h, xintxb1h)
        tmp = jnp.reshape(val, [M_Omega, M-M_Omega])
        theta = theta.at[0:M_Omega, 3*M_Omega:].set(tmp)
        theta = theta.at[3*M_Omega:, 0:M_Omega].set(tmp.T)

        val = vmap(lambda x1, x2, y1, y2: kernel.D_x2_kappa(x1, x2, y1, y2, params))(xintxb0v, xintxb1v, xintxb0h, xintxb1h)
        tmp = jnp.reshape(val, [M_Omega, M-M_Omega])
        theta = theta.at[M_Omega:2*M_Omega, 3*M_Omega:].set(tmp)
        theta = theta.at[3*M_Omega:, M_Omega:2*M_Omega].set(tmp.T)

        val = vmap(lambda x1, x2, y1, y2: kernel.Delta_x_kappa(x1, x2, y1, y2, params))(xintxb0v, xintxb1v, xintxb0h, xintxb1h)
        tmp = jnp.reshape(val, [M_Omega, M-M_Omega])
        theta = theta.at[2*M_Omega:3*M_Omega, 3*M_Omega:].set(tmp)
        theta = theta.at[3*M_Omega:, 2*M_Omega:3*M_Omega].set(tmp.T)

        return theta

    # build Nuggets
    @partial(jax.jit, static_argnums=(0, 2, 3))
    def build_nuggets(self, theta, M, M_Omega):
        trace11 = jnp.trace(theta[0:M_Omega, 0:M_Omega])
        trace22 = jnp.trace(theta[M_Omega:2*M_Omega, M_Omega:2*M_Omega])
        trace33 = jnp.trace(theta[2*M_Omega:3*M_Omega, 2*M_Omega:3*M_Omega])
        trace44 = jnp.trace(theta[3*M_Omega:, 3*M_Omega:])
        ratio = [trace11 / trace44, trace22 / trace44, trace33 / trace44]
        r_diag = jnp.concatenate((ratio[0] * jnp.ones((1, M_Omega)), ratio[1] * jnp.ones((1, M_Omega)), ratio[2] * jnp.ones((1, M_Omega)), jnp.ones((1, M-M_Omega))), axis=1)
        r = jnp.diag(r_diag[0])
        return r
    
    def trainGN(self, init_params, cfg):
        error = 1
        it = 0

        params = init_params
        u_kernel = self.kernel_generator()
        zk = jnp.zeros([2*self.M_Omega, ])  # represent the values of operators at the samples
        while  it < cfg.epoch: #error > cfg.tol and iter < cfg.epoch:
            # The optimization problem at each GN iteration is formulated as the following optimization problem
            # (Jz + y)^T\Theta^{-1}(Jz + y).
            # The minimization problem admits an explicit solution
            # z = -(J^T\Theta^{-1}J)^{-1}J^T\Theta^{-1}y

            # zk_current represents the values of operators of the functions at the previous iterations
            # evaluated at samples in the current iteration
            zk_current = zk
            zl_zeros = jnp.zeros([2*self.M_Omega, ])

            y = self.prolong(zl_zeros, zk_current, self.M_Omega)
            J = jacfwd(self.prolong)(zl_zeros, zk_current, self.M_Omega)
            
            # version 3: randomly sample the validation points but compute zk_at_valid inside the loop
            self.batch_size = cfg.batch_size
            
            # @jax.jit
            def solve(params):
                theta_u = self.build_theta(u_kernel, self.samples, params)
                nuggets_u = self.build_nuggets(theta_u, self.M, self.M_Omega)
                gram_u = theta_u + cfg.nugget * nuggets_u
                cho = jsp.linalg.cho_factor(gram_u)

                b = -jnp.dot(J.T, jsp.linalg.cho_solve(cho, y))
                tmp = jsp.linalg.cho_solve(cho, J)
                mtx = jnp.matmul(J.T, tmp)
                new_z = jnp.linalg.solve(mtx, b)

                new_uz = self.prolong(new_z, zk_current, self.M_Omega)

                u_weights = jsp.linalg.cho_solve(cho, new_uz)
                res = self.nonlinear_residual(new_z, zk_current, self.M_Omega)
                # res = self.nonlinear_residual(new_z, self.M_Omega)
                error = jnp.dot(res, res)
                return error, u_weights, new_z
            
            error, _, new_z = solve(params)

            it = it + 1

            # print(f"GN Epoch {iter}, param {params}, PDE Error {error}")
            print(f"GN Epoch {it}, PDE Error {error}")
            # print(jnp.exp(params["log_lphi"]), jnp.exp(params["log_lt"]))
            # print(params["lt"], params["lx"])
            

            ##### Make prediction #####
            data = sio.loadmat("./data/eikonal_reference_v2.mat")
            XX = jnp.array(data["XX"])
            YY = jnp.array(data["YY"])
            u_ref = jnp.array(data["u_ref"])
            X_test = jnp.concatenate([XX.reshape(-1,1), YY.reshape(-1,1)], axis=-1)

            out = self.ls_net.apply(params, X_test)
            out = out.reshape(XX.shape)

            sio.savemat("./outputs/l_xy.mat", {"XX": np.array(XX), "YY": np.array(YY), "l": np.array(out)})

            plt.figure()
            plt.pcolormesh(XX, YY, out, cmap=colormaps["jet"])
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("length scale for x/y")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig("./figures/lxy_used.png")
            plt.close()

            u_func = self.gen_u(params, new_z, new_z, jnp.array([cfg.nugget]))
            u_pred = vmap(u_func)(X_test)
            u_pred = u_pred.reshape(XX.shape)

            diff = jnp.abs(u_pred - u_ref)
            print("L infinity error:", jnp.max(diff))
            print("L2 error:", jnp.sqrt(jnp.mean(diff**2)), flush=True)

            plt.figure()
            plt.pcolormesh(XX, YY, u_pred, cmap=colormaps["jet"])
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("GP")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig("./figures/u_pred_test.png")
            plt.close()

            zk = jax.lax.stop_gradient(new_z)

        u_func = self.gen_u(params, zk, zk, jnp.array([cfg.nugget]))
        return u_func
