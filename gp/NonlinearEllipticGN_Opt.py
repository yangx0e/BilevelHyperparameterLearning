from utilities.domains import *
import numpy as np
import jax.numpy as jnp
from jax import grad, hessian, jacfwd, value_and_grad, jit, lax
from jax import vmap
from jax import config
from jax.scipy.linalg import solve_triangular
import jax.scipy as jsp
import optax
config.update("jax_enable_x64", True)
np.set_printoptions(precision=20)

class NonlinearElliptic_GN(object):
    # the equation is -Delta u + alpha * (u ** m) = f
    def __init__(self, kernel_generator, domain, alpha, m):
        self.kernel_generator = kernel_generator
        self.domain = domain
        self.alpha = alpha
        self.m = m

    def sampling(self, cfg):
        self.samples = self.domain.sampling(cfg.M, cfg.M_Omega)
        self.M = cfg.M
        self.M_Omega = cfg.M_Omega
        self.prepare_data()

    def set_samples(self, M, M_Omega, samples):
        self.samples = samples
        self.M = M
        self.M_Omega = M_Omega
        self.prepare_data()

    def prepare_data(self):
        # compute the values of f in the interior
        self.fxomega = vmap(self.f)(self.samples[:self.M_Omega, 0], self.samples[:self.M_Omega, 1])
        self.gbdr = np.zeros(self.M - self.M_Omega)

    # define the exact solution
    def u(self, x1, x2):
        return jnp.sin(jnp.pi * x1) * jnp.sin(jnp.pi * x2) + 4 * jnp.sin(4 * jnp.pi * x1) * jnp.sin(4 * jnp.pi * x2)

    # define the source term
    def f(self, x1, x2):
        return -grad(grad(self.u, 0), 0)(x1, x2) - grad(grad(self.u, 1), 1)(x1, x2) + self.alpha * (self.u(x1, x2) ** self.m)

    def residual(self, z, M, M_Omega, uk_func, samples):
        delta_v = z[:M_Omega]
        v = z[M_Omega:]

        uk = vmap(uk_func)(samples)
        f = vmap(self.f)(samples[:, 0], samples[:, 1])
        return -delta_v + 3 * uk ** 2 * v - 2 * uk ** 3 - f

    def nonlinear_residual(self, z, M, M_Omega, uk_func, samples):
        delta_v = z[:M_Omega]
        v = z[M_Omega:]

        uk = vmap(uk_func)(samples)
        f = vmap(self.f)(samples[:, 0], samples[:, 1])
        # return -delta_v + 3 * uk ** 2 * v - 2 * uk ** 3 - f
        return -delta_v + v ** 3 - f

    def gen_uk(self, params, z, nugget):
        kernel = self.kernel_generator(params)
        theta = self.build_theta(kernel, self.samples, self.M, self.M_Omega)
        nuggets = self.build_nuggets(theta, self.M, self.M_Omega)
        gram = theta + nugget * nuggets
        cho = jsp.linalg.cho_factor(gram)

        zz = jnp.append(z, self.gbdr)
        w = jsp.linalg.cho_solve(cho, zz)
        return lambda x: jnp.dot(self.KxPhi(kernel, x), w)

    def KxPhi(self, kernel, nx):
        x = self.samples
        xint0 = x[:self.M_Omega, 0]
        xint1 = x[:self.M_Omega, 1]
        x0 = x[:, 0]
        x1 = x[:, 1]

        kxphi = jnp.zeros(self.M + self.M_Omega)
        val0 = vmap(lambda y1, y2: kernel.kappa(nx[0], nx[1], y1, y2))(x0, x1)
        kxphi = kxphi.at[self.M_Omega:self.M + self.M_Omega].set(val0)

        val1 = vmap(lambda y1, y2: kernel.Delta_x_kappa(nx[0], nx[1], y1, y2))(xint0, xint1)
        kxphi = kxphi.at[0:self.M_Omega].set(val1)

        return kxphi

    def DeltaKxPhi(self, kernel, nx):
        x = self.samples
        xint0 = x[:self.M_Omega, 0]
        xint1 = x[:self.M_Omega, 1]
        x0 = x[:, 0]
        x1 = x[:, 1]

        Deltakxphi = jnp.zeros(self.M + self.M_Omega)
        val0 = vmap(lambda y1, y2: kernel.Delta_x_kappa(nx[0], nx[1], y1, y2))(x0, x1)
        Deltakxphi = Deltakxphi.at[self.M_Omega:self.M + self.M_Omega].set(val0)

        val1 = vmap(lambda y1, y2: kernel.Delta_x_Delta_y_kappa(nx[0], nx[1], y1, y2))(xint0, xint1)
        Deltakxphi = Deltakxphi.at[0:self.M_Omega].set(val1)

        return Deltakxphi

    def safe_cholesky(self, K, nuggets, initial_nugget=1e-13):
        def body_fun(val):
            nugget, L = val
            L = jnp.linalg.cholesky(K + nugget * nuggets)
            return nugget * 10, L  # Scale nugget if necessary

        def cond_fun(val):
            nugget, L = val
            return jnp.any(jnp.isnan(L))  # Continue if L has NaNs

        _, L = lax.while_loop(cond_fun, body_fun, (initial_nugget, jnp.linalg.cholesky(K + initial_nugget * nuggets)))
        return L


    def trainGN(self, init_params, cfg, params_reg_func = None):
        error = 1
        iter = 0
        self.M = cfg.M
        self.M_Omega = cfg.M_Omega

        zl = np.random.rand(2 * self.M_Omega)

        params = init_params
        nugget = jnp.log(jnp.exp(params[-1]) + 1)
        uk_func = self.gen_uk(params[:-1], zl, nugget)

        while error > cfg.tol and iter < cfg.epoch:
            zl_zeros = jnp.zeros(2 * self.M_Omega)
            y = -self.residual(zl_zeros, self.M, self.M_Omega, uk_func, self.samples[:self.M_Omega, :])
            J = jacfwd(self.residual)(zl_zeros, self.M, self.M_Omega, uk_func, self.samples[:self.M_Omega, :])

            def loss_res_learning_params(params, valid_samples, M, M_Omega):
                kernel_params = params[:-1]
                nugget = jnp.log(jnp.exp(params[-1]) + 1)
                u_kernel = self.kernel_generator(kernel_params)
                theta = self.build_theta(u_kernel, self.samples, self.M, self.M_Omega)
                nuggets = self.build_nuggets(theta, self.M, self.M_Omega)
                gram = theta + nugget * nuggets
                L = jnp.linalg.cholesky(gram)
                # L = self.safe_cholesky(theta, nuggets, nugget)
                P = jnp.concatenate((jnp.eye(2 * self.M_Omega), jnp.zeros((self.M - self.M_Omega, 2 * self.M_Omega))), axis=0)
                LinvP = jsp.linalg.solve_triangular(L, P, lower=True)
                mtx = cfg.lbda * jnp.matmul(LinvP.T, LinvP) + jnp.matmul(J.T, J)
                cho = jsp.linalg.cho_factor(mtx)
                new_z = jsp.linalg.cho_solve(cho, jnp.matmul(J.T, y))

                zz = jnp.concatenate((new_z, self.gbdr))
                u_weights = jsp.linalg.solve_triangular(L.T, jsp.linalg.solve_triangular(L, zz, lower=True), lower=False)

                u_vals, delta_u_vals = self.eval_u_ops(u_kernel, u_weights, valid_samples)

                res = self.nonlinear_residual(jnp.concatenate((delta_u_vals, u_vals)), M, M_Omega, uk_func, valid_samples)
                return res

            if cfg.learning_method == "adam":
                def loss_learning_params(params, valid_samples, M, M_Omega):
                    res = loss_res_learning_params(params, valid_samples, M, M_Omega)
                    return jnp.dot(res, res)# + cfg.reg * params_reg_func(params)

                optimizer = optax.adam(learning_rate=cfg.learning_lr)
                opt_state = optimizer.init(params)
                epoch = 0
                while epoch < cfg.learning_epoch:
                    valid_samples = self.domain.sampling(cfg.valid_M, cfg.valid_M_Omega)
                    valid_samples = valid_samples[:cfg.valid_M_Omega, :]

                    current_loss, grads = value_and_grad(loss_learning_params)(params, valid_samples, cfg.valid_M, cfg.valid_M_Omega)
                    # Update parameters and return new state
                    updates, pre_opt_state = optimizer.update(grads, opt_state, params)
                    pre_params = optax.apply_updates(params, updates)

                    if jnp.any(jnp.isnan(pre_params)):
                        nugget = jnp.log(jnp.exp(params[-1]) + 1) * 2
                        params = params.at[-1].set(jnp.log(jnp.exp(nugget) - 1))
                        print(f"nugget is too small, the new nugget is {nugget}")
                        continue

                    params = pre_params
                    opt_state = pre_opt_state

                    nugget = jnp.log(jnp.exp(params[-1]) + 1)
                    # print(f"        Learning Hyperparameters, GN Epoch {iter}, Learning Hyper Epoch {epoch}, Current Loss {current_loss}, Current nugget {nugget}")
                    print(
                        f"          Learning Hyperparameters, GN Epoch {iter}, Learning Hyper Epoch {epoch}, Current Loss {current_loss}, Current parameter {params}")

                    if current_loss <= cfg.learning_tol:
                        print(f"Stopping early at epoch {epoch} due to loss threshold")
                        break

                    epoch = epoch + 1

            elif cfg.learning_method == "newton":
                valid_samples = self.domain.sampling(cfg.valid_M, cfg.valid_M_Omega)
                valid_samples = valid_samples[:cfg.valid_M_Omega, :]
                gn_error = 1
                gn_iter = 0
                def loss_learning_params(params, valid_samples, M, M_Omega):
                    res = loss_res_learning_params(params, valid_samples, M, M_Omega)
                    return jnp.dot(res, res)# + cfg.reg * params_reg_func(params)

                while gn_error > cfg.learning_tol and gn_iter < cfg.learning_epoch:
                    grad_f = grad(loss_learning_params)(params, valid_samples, cfg.valid_M, cfg.valid_M_Omega)
                    hess_f = hessian(loss_learning_params)(params, valid_samples, cfg.valid_M, cfg.valid_M_Omega)
                    gn_delta = 0
                    A = hess_f
                    b = -grad_f
                    if jnp.ndim(A) == 0:
                        gn_delta = b / A
                    else:
                        gn_delta = jnp.linalg.lstsq(A, b, rcond=None)[0]
                    params = params + cfg.learning_lr * gn_delta

                    params = params + cfg.learning_lr * gn_delta
                    res = loss_res_learning_params(params, valid_samples, cfg.valid_M, cfg.valid_M_Omega)
                    gn_error = jnp.dot(gn_delta, gn_delta)
                    current_loss = jnp.dot(res, res)

                    nugget = jnp.log(jnp.exp(params[-1]) + 1)
                    print(f"        Learning Hyperparameters, GN Epoch {iter}, Learning Hyper Epoch {gn_iter}, Current Loss {current_loss}, Current Step Norm {gn_error}, Current nugget {nugget}")

                    if current_loss <= cfg.learning_tol:
                        print(f"Stopping early at epoch {gn_iter} due to loss threshold")
                        break
                    gn_iter = gn_iter + 1
            else:
                valid_samples = self.domain.sampling(cfg.valid_M, cfg.valid_M_Omega)
                valid_samples = valid_samples[:cfg.valid_M_Omega, :]
                gn_error = 1
                gn_iter = 0
                while gn_error > cfg.learning_tol and gn_iter < cfg.learning_epoch:
                    gn_y = -loss_res_learning_params(params, valid_samples, cfg.valid_M, cfg.valid_M_Omega)
                    gn_J = jacfwd(loss_res_learning_params)(params, valid_samples, cfg.valid_M, cfg.valid_M_Omega)
                    gn_delta = 0
                    A = gn_J.T @ gn_J
                    b = gn_J.T @ gn_y
                    if jnp.ndim(A) == 0:
                        gn_delta = b / A
                    else:
                        gn_delta = jnp.linalg.lstsq(A, b, rcond=None)[0]

                    params = params + cfg.learning_lr * gn_delta
                    res = loss_res_learning_params(params, valid_samples, cfg.valid_M, cfg.valid_M_Omega)
                    gn_error = jnp.dot(gn_delta, gn_delta)
                    current_loss = jnp.dot(res, res)

                    nugget = jnp.log(jnp.exp(params[-1]) + 1)
                    print(
                        f"        Learning Hyperparameters, GN Epoch {iter}, Learning Hyper Epoch {gn_iter}, Current Loss {current_loss}, Current Step Norm {gn_error}, Current nugget {nugget}")

                    if current_loss <= cfg.learning_tol:
                        print(f"Stopping early at epoch {gn_iter} due to loss threshold")
                        break
                    gn_iter = gn_iter + 1

            iter = iter + 1

            nugget = jnp.log(jnp.exp(params[-1]) + 1)

            u_kernel = self.kernel_generator(params[:-1])
            theta = self.build_theta(u_kernel, self.samples, self.M, self.M_Omega)
            nuggets = self.build_nuggets(theta, self.M, self.M_Omega)
            # gram = theta + nugget * nuggets
            # L = jnp.linalg.cholesky(gram)
            L = self.safe_cholesky(theta, nuggets, nugget)
            P = jnp.concatenate((jnp.eye(2 * self.M_Omega), jnp.zeros((self.M - self.M_Omega, 2 * self.M_Omega))),
                                axis=0)
            LinvP = jsp.linalg.solve_triangular(L, P, lower=True)
            mtx = cfg.lbda * jnp.matmul(LinvP.T, LinvP) + jnp.matmul(J.T, J)
            cho = jsp.linalg.cho_factor(mtx)
            new_z = jsp.linalg.cho_solve(cho, jnp.matmul(J.T, y))


            # print(f"Epoch {iter}, Error {error}, Optimal parameter {params}")
            #print(f"Epoch {iter}, Error {error}")

            uk_func = self.gen_uk(params[:-1], new_z, nugget)

        return uk_func

    def Knm(self, kernel, xl, Ml, M_Omegal, xr, Mr, M_Omegar):
        theta = jnp.zeros((Ml + M_Omegal, Mr + M_Omegar))

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

        val = vmap(lambda x1, x2, y1, y2: kernel.Delta_x_Delta_y_kappa(x1, x2, y1, y2))(xlintxrint0v, xlintxrint1v,
                                                                                             xlintxrint0h, xlintxrint1h)
        theta = theta.at[0:M_Omegal, 0:M_Omegar].set(jnp.reshape(val, (M_Omegal, M_Omegar)))

        val = vmap(lambda x1, x2, y1, y2: kernel.Delta_x_kappa(x1, x2, y1, y2))(xlintxr0v, xlintxr1v, xlintxr0h, xlintxr1h)
        theta = theta.at[0:M_Omegal, M_Omegar:].set(jnp.reshape(val, (M_Omegal, Mr)))

        val = vmap(lambda x1, x2, y1, y2: kernel.Delta_y_kappa(x1, x2, y1, y2))(xlxrint0v, xlxrint1v, xlxrint0h, xlxrint1h)
        theta = theta.at[M_Omegal:, 0:M_Omegar].set(jnp.reshape(val, (Ml, M_Omegar)))

        val = vmap(lambda x1, x2, y1, y2: kernel.kappa(x1, x2, y1, y2))(xlxr0v, xlxr1v, xlxr0h, xlxr1h)
        theta = theta.at[M_Omegal:, M_Omegar:].set(jnp.reshape(val, (Ml, Mr)))

        return theta

    def build_theta(self, kernel, x, M, M_Omega):
        return self.Knm(kernel, x, M, M_Omega, x, M, M_Omega)

    # build Nuggets
    def build_nuggets(self, theta, M, M_Omega):
        trace11 = jnp.trace(theta[0:M_Omega, 0:M_Omega])
        trace22 = jnp.trace(theta[M_Omega:M + M_Omega, M_Omega:M + M_Omega])
        ratio = trace11 / trace22
        r_diag = jnp.concatenate((ratio * jnp.ones((1, M_Omega)), jnp.ones((1, M))), axis=1)
        r = jnp.diag(r_diag[0])
        return r

    def eval_u_ops(self, kernel, weights, nx):
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

        Theta_u_test = jnp.zeros((nxl, self.M_Omega + self.M))

        # constructing Theta_u_test matrix
        val = vmap(lambda x1, x2, y1, y2: kernel.Delta_y_kappa(x1, x2, y1, y2))(xxint0v, xxint1v, xxint0h,
                                                                                       xxint1h)
        Theta_u_test = Theta_u_test.at[:, :self.M_Omega].set(jnp.reshape(val, (nxl, self.M_Omega)))

        val = vmap(lambda x1, x2, y1, y2: kernel.kappa(x1, x2, y1, y2))(xx0v, xx1v, xx0h, xx1h)
        Theta_u_test = Theta_u_test.at[:, self.M_Omega:].set(jnp.reshape(val, (nxl, self.M)))

        u_vals = jnp.matmul(Theta_u_test, weights)

        # Calculate Delta_u
        Theta_u_test = jnp.zeros((nxl, self.M_Omega + self.M))

        # constructing Theta_u_test matrix
        val = vmap(lambda x1, x2, y1, y2: kernel.Delta_x_Delta_y_kappa(x1, x2, y1, y2))(xxint0v, xxint1v, xxint0h,
                                                                                            xxint1h)
        Theta_u_test = Theta_u_test.at[:, :self.M_Omega].set(jnp.reshape(val, (nxl, self.M_Omega)))

        val = vmap(lambda x1, x2, y1, y2: kernel.Delta_x_kappa(x1, x2, y1, y2))(xx0v, xx1v, xx0h, xx1h)
        Theta_u_test = Theta_u_test.at[:, self.M_Omega:].set(jnp.reshape(val, (nxl, self.M)))

        Delta_u_vals = jnp.matmul(Theta_u_test, weights)

        return u_vals, Delta_u_vals

