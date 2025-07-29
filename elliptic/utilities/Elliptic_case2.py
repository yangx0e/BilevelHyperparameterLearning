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


class Elliptic(object):
    # Nonlinear elliptic equation: -Delta u + alpha u^m = f

    def __init__(self, kernel_generator, domain):
        self.kernel_generator = kernel_generator
        self.domain = domain

        self.alpha = 1
        self.m = 3

    def u_fn(self, x1, x2):
        return jnp.sin(jnp.pi * x1) * jnp.sin(jnp.pi * x2) + 4 * jnp.sin(
            4 * jnp.pi * x1
        ) * jnp.sin(4 * jnp.pi * x2)

    # define the source term
    def f_fn(self, x1, x2):
        return (
            -grad(grad(self.u_fn, 0), 0)(x1, x2)
            - grad(grad(self.u_fn, 1), 1)(x1, x2)
            + self.alpha * (self.u_fn(x1, x2) ** self.m)
        )

    def sampling(self, cfg):
        # self.samples = self.domain.sampling(cfg.M, cfg.M_Omega)
        # self.M = cfg.M
        # self.M_Omega = cfg.M_Omega
        # self.samples = jnp.array(self.samples)
        # self.gbdr = jnp.zeros([cfg.M - cfg.M_Omega])
        # self.f = vmap(self.f_fn)(
        #     self.samples[:cfg.M_Omega, 0], self.samples[:cfg.M_Omega, 1]
        # )

        # self.samples_valid = self.domain.sampling(cfg.valid_M, cfg.valid_M_Omega)
        # self.samples_interior_valid = self.samples_valid[:cfg.valid_M]
        # self.gbdr_valid = jnp.zeros([cfg.valid_M - cfg.valid_M_Omega])

        # sio.savemat(
        #     "./data/elliptic.mat",
        #     {
        #         "samples": self.samples,
        #         "M": cfg.M, "M_Omega": cfg.M_Omega,
        #         "samples_valid": self.samples_valid,
        #         "samples_interior_valid": self.samples_interior_valid,
        #         "valid_M": cfg.valid_M, "valid_M_Omega": cfg.valid_M_Omega,
        #     }
        # )

        data = sio.loadmat("./data/elliptic.mat")
        self.samples = data["samples"]
        # self.samples_valid = data["samples_valid"]
        self.samples_interior_valid = data["samples_interior_valid"]
        self.M = data["M"].astype(np.int32).reshape([])
        self.M_Omega = data["M_Omega"].astype(np.int32).reshape([])
        self.gbdr = jnp.zeros([self.M - self.M_Omega])
        self.f = vmap(self.f_fn)(
            self.samples[:self.M_Omega, 0], self.samples[:self.M_Omega, 1]
        )
        # self.samples_interior_valid = self.samples_interior_valid
        valid_M = data["valid_M"].astype(np.int32).reshape([])
        valid_M_Omega = data["valid_M_Omega"].astype(np.int32).reshape([])
        self.gbdr_valid = jnp.zeros([valid_M - valid_M_Omega, ])

    def sampling_test(self):

        data = sio.loadmat("./data/elliptic.mat")
        M_Omega = data["M_Omega"].astype(np.int32).reshape([])
        M = data["M"].astype(np.int32).reshape([])
        valid_M_Omega = data["valid_M_Omega"].astype(np.int32).reshape([])

        _samples = data["samples"]
        _samples_interior = _samples[:M_Omega]
        _samples_bdry = _samples[M_Omega:]
        _samples_interior_valid = data["samples_interior_valid"]
        self.samples = jnp.concatenate(
            [_samples_interior, _samples_interior_valid, _samples_bdry], axis=0,
        )
        self.M = M + valid_M_Omega
        self.M_Omega = M_Omega + valid_M_Omega

        self.gbdr = jnp.zeros([self.M - self.M_Omega])
        self.f = vmap(self.f_fn)(
            self.samples[:self.M_Omega, 0], self.samples[:self.M_Omega, 1]
        )

    def residual(self, z, zk, M_Omega, samples):
        # Delta u, u
        delta_u = z[0:M_Omega]
        u = z[M_Omega:2*M_Omega]

        uk = zk[M_Omega:2*M_Omega]

        f = vmap(self.f_fn)(samples[:, 0], samples[:, 1])
        out = -delta_u + 3 * uk**2 * u - 2 * uk**3 - f
        return out

    def prolong(self, z, zk):
        # this is for training
        # u
        u = z
        uk = zk

        delta_u = 3 * uk**2 * u - 2 * uk**3 - self.f
        return jnp.concatenate([delta_u, u, self.gbdr])

    def nonlinear_residual(self, z, zk):
        # this is for training
        u = z
        uk = zk

        delta_u = 3 * uk**2 * u - 2 * uk**3 - self.f

        out = -delta_u + u**3 - self.f
        return out

    def gen_u(self, params, z, zk, cfg):
        kernel = self.kernel_generator()
        theta = self.build_theta(kernel, self.samples, self.M, self.M_Omega, params)
        nuggets = self.build_nuggets(theta, self.M, self.M_Omega)
        gram = theta + cfg.nugget * nuggets
        cho = jsp.linalg.cho_factor(gram)

        uz = self.prolong(z, zk)

        u_weights = jsp.linalg.cho_solve(cho, uz)
        return lambda x: jnp.dot(self.KxPhi(kernel, x, params), u_weights)

    def KxPhi(self, kernel, nx, params):
        x = self.samples
        xint0 = x[: self.M_Omega, 0]
        xint1 = x[: self.M_Omega, 1]
        xb0 = x[self.M_Omega :, 0]
        xb1 = x[self.M_Omega :, 1]

        kxphi = jnp.zeros(
            [
                2 * self.M_Omega + self.M - self.M_Omega,
            ]
        )

        val = vmap(lambda y1, y2: kernel.Delta_y_kappa(nx[0], nx[1], y1, y2, params))(
            xint0, xint1
        )
        kxphi = kxphi.at[0 : self.M_Omega].set(val)

        val = vmap(lambda y1, y2: kernel.kappa(nx[0], nx[1], y1, y2, params))(
            xint0, xint1
        )
        kxphi = kxphi.at[self.M_Omega : 2 * self.M_Omega].set(val)

        val = vmap(lambda y1, y2: kernel.kappa(nx[0], nx[1], y1, y2, params))(xb0, xb1)
        kxphi = kxphi.at[2 * self.M_Omega :].set(val)

        return kxphi

    def build_theta(self, kernel, x, M, M_Omega, params):
        # delta_u, u, u_b
        theta = jnp.zeros([2 * M_Omega + M - M_Omega, 2 * M_Omega + M - M_Omega])

        xint0 = jnp.reshape(x[:M_Omega, 0], [M_Omega, 1])
        xint1 = jnp.reshape(x[:M_Omega, 1], [M_Omega, 1])
        xb0 = jnp.reshape(x[M_Omega:, 0], [M - M_Omega, 1])
        xb1 = jnp.reshape(x[M_Omega:, 1], [M - M_Omega, 1])

        # bdr v.s. bdr
        xxb0v = jnp.tile(xb0, M - M_Omega).flatten()
        xxb0h = jnp.tile(jnp.transpose(xb0), [M - M_Omega, 1]).flatten()
        xxb1v = jnp.tile(xb1, M - M_Omega).flatten()
        xxb1h = jnp.tile(jnp.transpose(xb1), [M - M_Omega, 1]).flatten()

        # int v.s. bdr
        xintxb0v = jnp.tile(xint0, M - M_Omega).flatten()
        xintxb0h = jnp.tile(jnp.transpose(xb0), [M_Omega, 1]).flatten()
        xintxb1v = jnp.tile(xint1, M - M_Omega).flatten()
        xintxb1h = jnp.tile(jnp.transpose(xb1), [M_Omega, 1]).flatten()

        # int v.s. int
        xintxint0v = jnp.tile(xint0, M_Omega).flatten()
        xintxint0h = jnp.tile(jnp.transpose(xint0), [M_Omega, 1]).flatten()
        xintxint1v = jnp.tile(xint1, M_Omega).flatten()
        xintxint1h = jnp.tile(jnp.transpose(xint1), [M_Omega, 1]).flatten()

        # delta_u, u, u_b
        val = vmap(
            lambda x1, x2, y1, y2: kernel.Delta_x_Delta_y_kappa(x1, x2, y1, y2, params)
        )(xintxint0v, xintxint1v, xintxint0h, xintxint1h)
        theta = theta.at[0:M_Omega, 0:M_Omega].set(jnp.reshape(val, [M_Omega, M_Omega]))

        val = vmap(lambda x1, x2, y1, y2: kernel.Delta_x_kappa(x1, x2, y1, y2, params))(
            xintxint0v, xintxint1v, xintxint0h, xintxint1h
        )
        tmp = np.reshape(val, [M_Omega, M_Omega])
        theta = theta.at[0:M_Omega, M_Omega : 2 * M_Omega].set(tmp)
        theta = theta.at[M_Omega : 2 * M_Omega, 0:M_Omega].set(tmp.T)

        val = vmap(lambda x1, x2, y1, y2: kernel.kappa(x1, x2, y1, y2, params))(
            xintxint0v, xintxint1v, xintxint0h, xintxint1h
        )
        theta = theta.at[M_Omega : 2 * M_Omega, M_Omega : 2 * M_Omega].set(
            jnp.reshape(val, [M_Omega, M_Omega])
        )

        val = vmap(lambda x1, x2, y1, y2: kernel.Delta_x_kappa(x1, x2, y1, y2, params))(
            xintxb0v, xintxb1v, xintxb0h, xintxb1h
        )
        tmp = jnp.reshape(val, [M_Omega, M - M_Omega])
        theta = theta.at[0:M_Omega, 2 * M_Omega :].set(tmp)
        theta = theta.at[2 * M_Omega :, 0:M_Omega].set(tmp.T)

        val = vmap(lambda x1, x2, y1, y2: kernel.kappa(x1, x2, y1, y2, params))(
            xintxb0v, xintxb1v, xintxb0h, xintxb1h
        )
        tmp = jnp.reshape(val, [M_Omega, M - M_Omega])
        theta = theta.at[M_Omega : 2 * M_Omega, 2 * M_Omega :].set(tmp)
        theta = theta.at[2 * M_Omega :, M_Omega : 2 * M_Omega].set(tmp.T)

        val = vmap(lambda x1, x2, y1, y2: kernel.kappa(x1, x2, y1, y2, params))(
            xxb0v, xxb1v, xxb0h, xxb1h
        )
        theta = theta.at[2 * M_Omega :, 2 * M_Omega :].set(
            jnp.reshape(val, [M - M_Omega, M - M_Omega])
        )

        return theta

    # build Nuggets
    def build_nuggets(self, theta, M, M_Omega):
        trace11 = jnp.trace(theta[0:M_Omega, 0:M_Omega])
        trace22 = jnp.trace(theta[M_Omega:, M_Omega:])
        ratio = [trace11 / trace22]
        r_diag = jnp.concatenate(
            (ratio[0] * jnp.ones([1, M_Omega]), jnp.ones([1, M])), axis=1
        )
        r = jnp.diag(r_diag[0])
        return r

    def eval_ops(self, kernel, u_weights, nx, params):
        x = self.samples
        xint0 = jnp.reshape(x[: self.M_Omega, 0], [self.M_Omega, 1])
        xint1 = jnp.reshape(x[: self.M_Omega, 1], [self.M_Omega, 1])
        xb0 = jnp.reshape(x[self.M_Omega :, 0], [self.M - self.M_Omega, 1])
        xb1 = jnp.reshape(x[self.M_Omega :, 1], [self.M - self.M_Omega, 1])

        nxl = len(nx)
        nx0 = jnp.reshape(nx[:, 0], (nxl, 1))
        nx1 = jnp.reshape(nx[:, 1], (nxl, 1))

        xxb0v = jnp.tile(nx0, self.M - self.M_Omega).flatten()
        xxb0h = jnp.tile(jnp.transpose(xb0), (nxl, 1)).flatten()
        xxb1v = jnp.tile(nx1, self.M - self.M_Omega).flatten()
        xxb1h = jnp.tile(jnp.transpose(xb1), (nxl, 1)).flatten()

        xxint0v = jnp.tile(nx0, self.M_Omega).flatten()
        xxint0h = jnp.tile(jnp.transpose(xint0), (nxl, 1)).flatten()
        xxint1v = jnp.tile(nx1, self.M_Omega).flatten()
        xxint1h = jnp.tile(jnp.transpose(xint1), (nxl, 1)).flatten()

        # Delta u, u, u_b
        # Delta u
        mtx = jnp.zeros([nxl, 2 * self.M_Omega + self.M - self.M_Omega])

        val = vmap(
            lambda x1, x2, y1, y2: kernel.Delta_x_Delta_y_kappa(x1, x2, y1, y2, params)
        )(xxint0v, xxint1v, xxint0h, xxint1h)
        mtx = mtx.at[:, 0 : self.M_Omega].set(jnp.reshape(val, [nxl, self.M_Omega]))

        val = vmap(lambda x1, x2, y1, y2: kernel.Delta_x_kappa(x1, x2, y1, y2, params))(
            xxint0v, xxint1v, xxint0h, xxint1h
        )
        mtx = mtx.at[:, self.M_Omega : 2 * self.M_Omega].set(
            jnp.reshape(val, [nxl, self.M_Omega])
        )

        val = vmap(lambda x1, x2, y1, y2: kernel.Delta_x_kappa(x1, x2, y1, y2, params))(
            xxb0v, xxb1v, xxb0h, xxb1h
        )
        mtx = mtx.at[:, 2 * self.M_Omega :].set(
            jnp.reshape(val, [nxl, self.M - self.M_Omega])
        )

        delta_u = mtx.dot(u_weights)

        # u
        mtx = jnp.zeros([nxl, 2 * self.M_Omega + self.M - self.M_Omega])

        val = vmap(lambda x1, x2, y1, y2: kernel.Delta_y_kappa(x1, x2, y1, y2, params))(
            xxint0v, xxint1v, xxint0h, xxint1h
        )
        mtx = mtx.at[:, 0 : self.M_Omega].set(jnp.reshape(val, [nxl, self.M_Omega]))

        val = vmap(lambda x1, x2, y1, y2: kernel.kappa(x1, x2, y1, y2, params))(
            xxint0v, xxint1v, xxint0h, xxint1h
        )
        mtx = mtx.at[:, self.M_Omega : 2 * self.M_Omega].set(
            jnp.reshape(val, [nxl, self.M_Omega])
        )

        val = vmap(lambda x1, x2, y1, y2: kernel.kappa(x1, x2, y1, y2, params))(
            xxb0v, xxb1v, xxb0h, xxb1h
        )
        mtx = mtx.at[:, 2 * self.M_Omega :].set(
            jnp.reshape(val, [nxl, self.M - self.M_Omega])
        )

        u = mtx.dot(u_weights)

        return delta_u, u

    def trainGN(self, init_params, cfg):
        error = 1
        it = 0

        params = init_params
        u_kernel = self.kernel_generator()
        uk_params = params
        uk_weights = jnp.zeros(
            [
                2 * self.M_Omega + self.M - self.M_Omega,
            ]
        )
        zk = jnp.zeros(
            [
                1 * self.M_Omega,
            ]
        )  # represent the values of operators at the samples

        # seed = np.random.randint(0, 9999999)
        # seed = 4733051
        # print("seed: ", seed)
        # np.random.seed(seed)
        params_history = [params]
        loss_history = []
        while it < cfg.epoch:  # error > cfg.tol and it < cfg.epoch:
            # The optimization problem at each GN itation is formulated as the following optimization problem
            # (Jz + y)^T\Theta^{-1}(Jz + y).
            # The minimization problem admits an explicit solution
            # z = -(J^T\Theta^{-1}J)^{-1}J^T\Theta^{-1}y

            # zk_current represents the values of operators of the functions at the previous itations
            # evaluated at samples in the current itation
            zk_current = zk
            zl_zeros = jnp.zeros(
                [
                    1 * self.M_Omega,
                ]
            )

            y = self.prolong(zl_zeros, zk_current)
            J = jacfwd(self.prolong)(zl_zeros, zk_current)

            # valid_samples = self.domain.sampling(cfg.valid_M, cfg.valid_M_Omega)
            # valid_M = cfg.valid_M
            # valid_M_Omega = cfg.valid_M_Omega
            idx = np.random.choice(
                self.samples_interior_valid.shape[0],
                self.samples_interior_valid.shape[0],
                replace=False,
            )[: cfg.batch_size]
            batch_samples_interior_valid = self.samples_interior_valid[idx, :]
            delta_uk_vals, uk_vals = self.eval_ops(
                u_kernel,
                uk_weights,
                batch_samples_interior_valid,
                uk_params,
            )
            zk_at_valid = jnp.concatenate([delta_uk_vals, uk_vals])
            zk_at_valid = jax.lax.stop_gradient(zk_at_valid)
            self.batch_size = cfg.batch_size

            # @jax.jit
            def loss_learning_params(params, zk_at_valid, batch_samples_interior_valid):
                # kernel_params = params[:-1]
                # nugget = jnp.log(jnp.exp(params[-1]) + 1)
                kernel_params = params
                nugget = cfg.nugget
                theta_u = self.build_theta(
                    u_kernel, self.samples, self.M, self.M_Omega, kernel_params
                )
                nuggets_u = self.build_nuggets(theta_u, self.M, self.M_Omega)
                gram_u = theta_u + nugget * nuggets_u
                cho = jsp.linalg.cho_factor(gram_u)

                b = -jnp.dot(J.T, jsp.linalg.cho_solve(cho, y))
                tmp = jsp.linalg.cho_solve(cho, J)
                mtx = jnp.matmul(J.T, tmp)
                new_z = jnp.linalg.solve(mtx, b)

                new_uz = self.prolong(new_z, zk_current)

                u_weights = jsp.linalg.cho_solve(cho, new_uz)

                delta_u_vals, u_vals = self.eval_ops(
                    u_kernel,
                    u_weights,
                    batch_samples_interior_valid,
                    kernel_params,
                )
                z_at_valid = jnp.concatenate([delta_u_vals, u_vals])

                res = self.residual(z_at_valid, zk_at_valid, self.batch_size, batch_samples_interior_valid)

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
                params_history += [params]
                loss_history += [current_loss]
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
                # kernel_params = params[:-1]
                # nugget = jnp.log(jnp.exp(params[-1]) + 1)
                kernel_params = params
                nugget = cfg.nugget
                theta_u = self.build_theta(
                    u_kernel, self.samples, self.M, self.M_Omega, kernel_params
                )
                nuggets_u = self.build_nuggets(theta_u, self.M, self.M_Omega)
                gram_u = theta_u + nugget * nuggets_u
                cho = jsp.linalg.cho_factor(gram_u)

                b = -jnp.dot(J.T, jsp.linalg.cho_solve(cho, y))
                tmp = jsp.linalg.cho_solve(cho, J)
                mtx = jnp.matmul(J.T, tmp)
                new_z = jnp.linalg.solve(mtx, b)

                new_uz = self.prolong(new_z, zk_current)

                u_weights = jsp.linalg.cho_solve(cho, new_uz)
                res = self.nonlinear_residual(new_z, zk_current)
                error = jnp.dot(res, res)
                return error, u_weights, new_z

            error, u_weights, new_z = solve(params)

            it = it + 1

            # print(f"GN Epoch {it}, param {params}, PDE Error {error}")
            print(f"GN Epoch {it}, PDE Error {error}")
            # print(jnp.exp(params["log_lphi"]), jnp.exp(params["log_lt"]))
            # print(params["lt"], params["lx"])

            ##### Make prediction #####
            # kernel_params = params[:-1]
            # nugget = jnp.log(jnp.exp(params[-1]) + 1)
            kernel_params = params.copy()
            nugget = cfg.nugget

            N_pts = 60
            xx = jnp.linspace(0, 1, N_pts)
            yy = jnp.linspace(0, 1, N_pts)
            XX, YY = jnp.meshgrid(xx, yy)
            X_test = jnp.concatenate([XX.reshape(-1, 1), YY.reshape(-1, 1)], axis=1)

            u_func = self.gen_u(kernel_params, new_z, new_z, cfg)
            u_pred = vmap(u_func)(X_test)
            u_pred = u_pred.reshape(XX.shape)

            u_ref = vmap(self.u_fn)(X_test[:, 0], X_test[:, 1])
            u_ref = u_ref.reshape(XX.shape)

            diff = jnp.abs(u_pred - u_ref)
            print("L infinity error: ", jnp.max(diff))
            
            print("Learned kernel parameters: ", kernel_params)
            print("Learned nugget: ", nugget)

            plt.figure()
            plt.pcolormesh(XX, YY, u_pred, cmap=colormaps["jet"])
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("GP")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig("./figures/u_pred.png")
            plt.close()

            plt.figure()
            plt.pcolormesh(XX, YY, u_ref, cmap=colormaps["jet"])
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Reference")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig("./figures/u_ref.png")
            plt.close()

            uk_params = jax.lax.stop_gradient(params)
            uk_weights = jax.lax.stop_gradient(u_weights)
            zk = jax.lax.stop_gradient(new_z)

        params_history = np.stack(params_history)
        loss_history = np.stack(loss_history)
        np.savetxt("./outputs/params_history_case2", params_history)
        np.savetxt("./outputs/loss_history_case2", loss_history)
        u_func = self.gen_u(kernel_params, zk, zk, cfg)
        return u_func
    
    def testGN(self, init_params, cfg):
        error = 1
        it = 0

        params = init_params
        u_kernel = self.kernel_generator()
        zk = jnp.zeros(
            [
                1 * self.M_Omega,
            ]
        )  # represent the values of operators at the samples

        while it < cfg.epoch:  # error > cfg.tol and it < cfg.epoch:
            zk_current = zk
            zl_zeros = jnp.zeros(
                [
                    1 * self.M_Omega,
                ]
            )

            y = self.prolong(zl_zeros, zk_current)
            J = jacfwd(self.prolong)(zl_zeros, zk_current)

            @jax.jit
            def solve(params):
                # kernel_params = params[:-1]
                # nugget = jnp.log(jnp.exp(params[-1]) + 1)
                kernel_params = params
                nugget = cfg.nugget
                theta_u = self.build_theta(
                    u_kernel, self.samples, self.M, self.M_Omega, kernel_params
                )
                nuggets_u = self.build_nuggets(theta_u, self.M, self.M_Omega)
                gram_u = theta_u + nugget * nuggets_u
                cho = jsp.linalg.cho_factor(gram_u)

                b = -jnp.dot(J.T, jsp.linalg.cho_solve(cho, y))
                tmp = jsp.linalg.cho_solve(cho, J)
                mtx = jnp.matmul(J.T, tmp)
                new_z = jnp.linalg.solve(mtx, b)

                new_uz = self.prolong(new_z, zk_current)

                u_weights = jsp.linalg.cho_solve(cho, new_uz)
                res = self.nonlinear_residual(new_z, zk_current)
                error = jnp.dot(res, res)
                return error, u_weights, new_z

            error, u_weights, new_z = solve(params)

            it = it + 1

            # print(f"GN Epoch {it}, param {params}, PDE Error {error}")
            print(f"GN Epoch {it}, PDE Error {error}")
            # print(jnp.exp(params["log_lphi"]), jnp.exp(params["log_lt"]))
            # print(params["lt"], params["lx"])

            ##### Make prediction #####
            # kernel_params = params[:-1]
            # nugget = jnp.log(jnp.exp(params[-1]) + 1)
            kernel_params = params
            nugget = cfg.nugget

            N_pts = 60
            xx = jnp.linspace(0, 1, N_pts)
            yy = jnp.linspace(0, 1, N_pts)
            XX, YY = jnp.meshgrid(xx, yy)
            X_test = jnp.concatenate([XX.reshape(-1, 1), YY.reshape(-1, 1)], axis=1)

            u_func = self.gen_u(kernel_params, new_z, new_z, cfg)
            u_pred = vmap(u_func)(X_test)
            u_pred = u_pred.reshape(XX.shape)

            u_ref = vmap(self.u_fn)(X_test[:, 0], X_test[:, 1])
            u_ref = u_ref.reshape(XX.shape)

            diff = jnp.abs(u_pred - u_ref)
            print("L infinity error: ", jnp.max(diff))
            print("L2 error: ", jnp.sqrt(jnp.mean(diff**2)))
            
            print("Learned kernel parameters: ", kernel_params)
            print("Learned nugget: ", nugget)

            plt.figure()
            plt.pcolormesh(XX, YY, u_pred, cmap=colormaps["jet"])
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("GP")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig("./figures/u_pred.png")
            plt.close()

            plt.figure()
            plt.pcolormesh(XX, YY, u_ref, cmap=colormaps["jet"])
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Reference")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig("./figures/u_ref.png")
            plt.close()

            zk = jax.lax.stop_gradient(new_z)

        u_func = self.gen_u(kernel_params, zk, zk, cfg)
        return u_func
