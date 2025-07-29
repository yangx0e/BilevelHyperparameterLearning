import jax.numpy as jnp
from jax import vmap, jit, jacfwd, value_and_grad
from utilities.domains import *
from jax import config
import jax.scipy as jsp
import jax
import optax
from functools import partial
config.update("jax_enable_x64", True)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import scipy.io as sio


np.set_printoptions(precision=20)


class Burgers_GP(object):
    # Burgers Equation: u_t+ alpha u u_x - nu u_xx=0

    def __init__(self, kernel_generator, domain, nu, ls_net):
        self.kernel_generator = kernel_generator
        self.domain = domain
        self.nu = nu
        self.ls_net = ls_net

    def sampling(self, cfg):
        self.samples = self.domain.sampling(cfg.M, cfg.M_Omega)
        self.M = cfg.M
        self.M_Omega = cfg.M_Omega
        self.samples = jnp.array(self.samples)
        self.prepare_data()

        samples_interior_valid = self.domain.sampling(cfg.valid_M, cfg.valid_M_Omega)[:cfg.valid_M_Omega, :]
        samples_interior = self.samples[:cfg.M_Omega, :]
        samples_bdry = self.samples[cfg.M_Omega:, :]
        self.samples = jnp.concatenate(
            [samples_interior, samples_interior_valid, samples_bdry], axis=0,
        )
        self.M = cfg.M + cfg.valid_M_Omega
        self.M_Omega = cfg.M_Omega + cfg.valid_M_Omega

        # sio.savemat(
        #     "./data/samples.mat",
        #     {
        #         "interior": np.array(self.samples[:self.M_Omega]),
        #         "boundary": np.array(self.samples[self.M_Omega:]),
        #     }
        # )

    def prepare_data(self):
        # compute the values of f in the interior
        self.fxomega = vmap(self.f)(self.samples[:self.M_Omega, 0], self.samples[:self.M_Omega, 1])
        self.gbdr = vmap(self.g)(self.samples[self.M_Omega:, 0], self.samples[self.M_Omega:, 1])

    # define the boundary term
    def g(self, x1, x2):
        return -jnp.sin(jnp.pi * x2) * (x1 == 0)

    # define the exact solution
    def u(self, x1, x2):
        return 0

    # define the source term
    def f(self, x1, x2):
        return 0

    def residual(self, z, zk, M_Omega):
        # z contains the values of v, which are the solutions at the current GN iterations
        # zk contains the values of u, which are the solutions at the previous GN iterations

        vt = z[0:M_Omega]  # vt
        vx = z[M_Omega:2 * M_Omega]  # vx
        vxx = z[2 * M_Omega:3 * M_Omega] # vxx
        v = z[3 * M_Omega:4 * M_Omega]  # v

        # ut = zk[0:M_Omega]  # ut
        ux = zk[M_Omega:2 * M_Omega]  # ux
        # uxx = zk[2 * M_Omega:3 * M_Omega] # uxx
        u = zk[3 * M_Omega:4 * M_Omega]  # u

        return vt + v * ux + u * vx - u * ux - self.nu * vxx

    @partial(jax.jit, static_argnums=(0, 3))
    def prolong(self, z, zk, M_Omega):
        vt = z[0:M_Omega]  # vt
        vx = z[M_Omega:2 * M_Omega]  # vx
        v = z[2 * M_Omega:3 * M_Omega]  # v

        # ut = zk[0:M_Omega]  # ut
        ux = zk[M_Omega:2 * M_Omega]  # ux
        u = zk[2 * M_Omega:3 * M_Omega]  # u

        vxx = (vt + v * ux + u * vx - u * ux) / self.nu

        return jnp.concatenate((vt, vx, vxx, v, self.gbdr))

    @partial(jax.jit, static_argnums=(0, 3))
    def nonlinear_residual(self, z, zk, M_Omega):
        # z contains the values of v, which are the solutions at the current GN iterations

        vt = z[0:M_Omega]  # vt
        vx = z[M_Omega:2 * M_Omega]  # vx
        v = z[2 * M_Omega:3 * M_Omega]  # v

        # ut = zk[0:M_Omega]  # ut
        ux = zk[M_Omega:2 * M_Omega]  # ux
        u = zk[2 * M_Omega:3 * M_Omega]  # u

        vxx = (vt + v * ux + u * vx - u * ux) / self.nu

        return vt + v * vx - self.nu * vxx
    
    def gen_u(self, params, z, zk, cfg):
        kernel = self.kernel_generator()
        theta = self.build_theta(kernel, self.samples, params)
        nuggets = self.build_nuggets(theta, self.M, self.M_Omega)
        gram = theta + cfg.nugget * nuggets
        cho = jsp.linalg.cho_factor(gram)

        uz = self.prolong(z, zk, self.M_Omega)

        u_weights = jsp.linalg.cho_solve(cho, uz)
        return lambda x: jnp.dot(self.KxPhi(kernel, x, params), u_weights)

    def KxPhi(self, kernel, nx, params):
        x = self.samples
        xint0 = x[:self.M_Omega, 0]
        xint1 = x[:self.M_Omega, 1]
        x0 = x[:, 0]
        x1 = x[:, 1]

        kxphi = jnp.zeros(self.M + 3 * self.M_Omega)

        val = vmap(lambda y1, y2: kernel.D_y1_kappa(nx[0], nx[1], y1, y2, params))(xint0, xint1)
        kxphi = kxphi.at[0:self.M_Omega].set(val)

        val = vmap(lambda y1, y2: kernel.D_y2_kappa(nx[0], nx[1], y1, y2, params))(xint0, xint1)
        kxphi = kxphi.at[self.M_Omega:2*self.M_Omega].set(val)

        val = vmap(lambda y1, y2: kernel.DD_y2_kappa(nx[0], nx[1], y1, y2, params))(xint0, xint1)
        kxphi = kxphi.at[2 * self.M_Omega:3 * self.M_Omega].set(val)

        val = vmap(lambda y1, y2: kernel.kappa(nx[0], nx[1], y1, y2, params))(x0, x1)
        kxphi = kxphi.at[3 * self.M_Omega:].set(val)

        return kxphi

    # build theta
    # x: the sample points
    # M: the number of points, including those on the boundary
    # M_Omega: the number of points int the interior
    @partial(jax.jit, static_argnums=(0, 1))
    def build_theta(self, kernel, x, params):
        M = self.M
        M_Omega = self.M_Omega
        theta = jnp.zeros((M + 3 * M_Omega, M + 3 * M_Omega))

        xint0 = jnp.reshape(x[:M_Omega, 0], (M_Omega, 1))
        xint1 = jnp.reshape(x[:M_Omega, 1], (M_Omega, 1))
        x0 = jnp.reshape(x[:, 0], (M, 1))
        x1 = jnp.reshape(x[:, 1], (M, 1))

        # int + bdr v.s. int + bdr
        xx0v = jnp.tile(x0, M).flatten()
        xx0h = jnp.tile(jnp.transpose(x0), (M, 1)).flatten()
        xx1v = jnp.tile(x1, M).flatten()
        xx1h = jnp.tile(jnp.transpose(x1), (M, 1)).flatten()

        # int + bdr v.s. int
        xxint0v = jnp.tile(x0, M_Omega).flatten()
        xxint0h = jnp.tile(jnp.transpose(xint0), (M, 1)).flatten()
        xxint1v = jnp.tile(x1, M_Omega).flatten()
        xxint1h = jnp.tile(jnp.transpose(xint1), (M, 1)).flatten()

        # int v.s. int + bdr
        xintx0v = jnp.tile(xint0, M).flatten()
        xintx0h = jnp.tile(jnp.transpose(x0), (M_Omega, 1)).flatten()
        xintx1v = jnp.tile(xint1, M).flatten()
        xintx1h = jnp.tile(jnp.transpose(x1), (M_Omega, 1)).flatten()

        # int v.s. int
        xintxint0v = jnp.tile(xint0, M_Omega).flatten()
        xintxint0h = jnp.tile(jnp.transpose(xint0), (M_Omega, 1)).flatten()
        xintxint1v = jnp.tile(xint1, M_Omega).flatten()
        xintxint1h = jnp.tile(jnp.transpose(xint1), (M_Omega, 1)).flatten()

        val = vmap(lambda x1, x2, y1, y2: kernel.D_x1_D_y1_kappa(x1, x2, y1, y2, params))(xintxint0v, xintxint1v, xintxint0h, xintxint1h)
        theta = theta.at[0:M_Omega, 0:M_Omega].set(jnp.reshape(val, (M_Omega, M_Omega)))

        val = vmap(lambda x1, x2, y1, y2: kernel.D_x1_D_y2_kappa(x1, x2, y1, y2, params))(xintxint0v, xintxint1v, xintxint0h, xintxint1h)
        tmp = jnp.reshape(val, (M_Omega, M_Omega))
        theta = theta.at[0:M_Omega, M_Omega:2*M_Omega].set(tmp)
        theta = theta.at[M_Omega:2*M_Omega, 0:M_Omega].set(tmp.T)

        val = vmap(lambda x1, x2, y1, y2: kernel.D_x1_DD_y2_kappa(x1, x2, y1, y2, params))(xintxint0v, xintxint1v, xintxint0h, xintxint1h)
        tmp = jnp.reshape(val, (M_Omega, M_Omega))
        theta = theta.at[0:M_Omega, 2 * M_Omega:3 * M_Omega].set(tmp)
        theta = theta.at[2 * M_Omega:3 * M_Omega, 0:M_Omega].set(tmp.T)

        val = vmap(lambda x1, x2, y1, y2: kernel.D_x2_D_y2_kappa(x1, x2, y1, y2, params))(xintxint0v, xintxint1v, xintxint0h, xintxint1h)
        theta = theta.at[M_Omega:2*M_Omega, M_Omega:2 * M_Omega].set(jnp.reshape(val, (M_Omega, M_Omega)))

        val = vmap(lambda x1, x2, y1, y2: kernel.D_x2_DD_y2_kappa(x1, x2, y1, y2, params))(xintxint0v, xintxint1v, xintxint0h, xintxint1h)
        tmp = jnp.reshape(val, (M_Omega, M_Omega))
        theta = theta.at[M_Omega:2*M_Omega, 2 * M_Omega:3 * M_Omega].set(tmp)
        theta = theta.at[2 * M_Omega:3 * M_Omega, M_Omega:2*M_Omega].set(tmp.T)

        val = vmap(lambda x1, x2, y1, y2: kernel.DD_x2_DD_y2_kappa(x1, x2, y1, y2, params))(xintxint0v, xintxint1v, xintxint0h, xintxint1h)
        theta = theta.at[2 * M_Omega:3 * M_Omega, 2 * M_Omega:3 * M_Omega].set(jnp.reshape(val, (M_Omega, M_Omega)))

        val = vmap(lambda x1, x2, y1, y2: kernel.kappa(x1, x2, y1, y2, params))(xx0v, xx1v, xx0h, xx1h)
        theta = theta.at[3 * M_Omega:, 3 * M_Omega:].set(jnp.reshape(val, (M, M)))

        val = vmap(lambda x1, x2, y1, y2: kernel.D_x1_kappa(x1, x2, y1, y2, params))(xintx0v, xintx1v, xintx0h, xintx1h)
        tmp = jnp.reshape(val, (M_Omega, M))
        theta = theta.at[0:M_Omega, 3 * M_Omega:].set(tmp)
        theta = theta.at[3 * M_Omega:, 0:M_Omega].set(tmp.T)

        val = vmap(lambda x1, x2, y1, y2: kernel.D_x2_kappa(x1, x2, y1, y2, params))(xintx0v, xintx1v, xintx0h, xintx1h)
        tmp = jnp.reshape(val, (M_Omega, M))
        theta = theta.at[M_Omega:2*M_Omega, 3 * M_Omega:].set(tmp)
        theta = theta.at[3 * M_Omega:, M_Omega:2*M_Omega].set(tmp.T)

        val = vmap(lambda x1, x2, y1, y2: kernel.DD_x2_kappa(x1, x2, y1, y2, params))(xintx0v, xintx1v, xintx0h, xintx1h)
        tmp = jnp.reshape(val, (M_Omega, M))
        theta = theta.at[2 * M_Omega:3 * M_Omega, 3 * M_Omega:].set(tmp)
        theta = theta.at[3 * M_Omega:, 2 * M_Omega:3 * M_Omega].set(tmp.T)

        return theta

    # build Nuggets
    @partial(jax.jit, static_argnums=(0, 2, 3))
    def build_nuggets(self, theta, M, M_Omega):
        trace11 = jnp.trace(theta[0:M_Omega, 0:M_Omega])
        trace22 = jnp.trace(theta[M_Omega:2 * M_Omega, M_Omega:2 * M_Omega])
        trace33 = jnp.trace(theta[2 * M_Omega:3 * M_Omega, 2 * M_Omega:3 * M_Omega])
        trace44 = jnp.trace(theta[3 * M_Omega:, 3 * M_Omega:])
        ratio = [trace11 / trace44, trace22 / trace44, trace33 / trace44]
        r_diag = jnp.concatenate((ratio[0] * jnp.ones((1, M_Omega)), ratio[1] * jnp.ones((1, M_Omega)), ratio[2] * jnp.ones((1, M_Omega)), jnp.ones((1, M))), axis=1)
        r = jnp.diag(r_diag[0])
        return r
    
    def trainGN(self, init_params, cfg, params_reg_func = None):
        error = 1
        it = 0

        params = init_params
        u_kernel = self.kernel_generator()
        zk = jnp.zeros((3 * self.M_Omega))  # represent the values of operators at the samples
        while  it < cfg.epoch: #error > cfg.tol and iter < cfg.epoch:
            # The optimization problem at each GN iteration is formulated as the following optimization problem
            # (Jz + y)^T\Theta^{-1}(Jz + y).
            # The minimization problem admits an explicit solution
            # z = -(J^T\Theta^{-1}J)^{-1}J^T\Theta^{-1}y

            # zk_current represents the values of operators of the functions at the previous iterations
            # evaluated at samples in the current iteration
            zk_current = zk
            zl_zeros = jnp.zeros(3 * self.M_Omega)

            y = self.prolong(zl_zeros, zk_current, self.M_Omega)
            J = jacfwd(self.prolong)(zl_zeros, zk_current, self.M_Omega)
            
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
            
            error, u_weights, new_z = solve(params)

            it = it + 1

            # print(f"GN Epoch {iter}, param {params}, PDE Error {error}")
            print(f"GN Epoch {it}, PDE Error {error}")
            # print(jnp.exp(params["log_lphi"]), jnp.exp(params["log_lt"]))
            # print(params["lt"], params["lx"])
            x = jnp.linspace(-1, 1, 60)
            t = jnp.linspace(0, 1, 60)
            tt, xx = jnp.meshgrid(t, x)
            tt = tt.reshape([-1, 1])
            xx = xx.reshape([-1, 1])

            out = self.ls_net.apply(params, jnp.concatenate([tt, xx], axis=1))
            lt, lx = jnp.split(out, 2, axis=-1)
            lt = lt.reshape([60, 60])
            lx = lx.reshape([60, 60])
            tt = tt.reshape([60, 60])
            xx = xx.reshape([60, 60])

            sio.savemat(
                "./outputs/l_xy.mat", 
                {
                    "xx": np.array(xx), 
                    "tt": np.array(tt), 
                    "lx": np.array(lx),
                    "lt": np.array(lt),
                },
            )

            plt.figure()
            plt.pcolormesh(tt, xx, lt, cmap=colormaps["jet"])
            plt.xlabel("t")
            plt.ylabel("x")
            plt.title("length scale for t")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig("./figures/lt.png")
            plt.close()

            plt.figure()
            plt.pcolormesh(tt, xx, lx, cmap=colormaps["jet"])
            plt.xlabel("t")
            plt.ylabel("x")
            plt.title("length scale for x")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig("./figures/lx.png")
            plt.close()

            u_func = self.gen_u(params, new_z, new_z, cfg)
            X_test = jnp.concatenate([tt.reshape(-1,1), xx.reshape(-1,1)], axis=-1)
            u_pred = vmap(u_func)(X_test)
            u_pred = u_pred.reshape([60, 60])

            # GP interpolation and test accuracy
            [Gauss_pts, weights] = np.polynomial.hermite.hermgauss(80)
            def u_truth(x1, x2):
                temp = x2-jnp.sqrt(4*cfg.nu*x1)*Gauss_pts
                val1 = weights * jnp.sin(jnp.pi*temp) * jnp.exp(-jnp.cos(jnp.pi*temp)/(2*jnp.pi*cfg.nu))
                val2 = weights * jnp.exp(-jnp.cos(jnp.pi*temp)/(2*jnp.pi*cfg.nu))
                return -jnp.sum(val1)/jnp.sum(val2)

            u_ref = vmap(u_truth)(X_test[:, 0], X_test[:, 1])
            u_ref = u_ref.reshape([60, 60])

            plt.figure()
            plt.pcolormesh(tt, xx, u_ref, cmap=colormaps["jet"])
            plt.xlabel("t")
            plt.ylabel("x")
            plt.title("Reference")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig("./figures/u_ref.png")
            plt.close()

            plt.figure()
            plt.pcolormesh(tt, xx, u_pred, cmap=colormaps["jet"])
            plt.xlabel("t")
            plt.ylabel("x")
            plt.title("GP")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig("./figures/u_pred.png")
            plt.close()

            err = jnp.abs(u_ref - u_pred)
            print("L infinity error: ", jnp.max(err))

            zk = jax.lax.stop_gradient(new_z)

        u_func = self.gen_u(params, zk, zk, cfg)
        return u_func

