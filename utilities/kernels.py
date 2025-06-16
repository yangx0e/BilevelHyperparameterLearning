#import torch
#from functorch import vmap, grad
import jax.numpy as jnp
from jax import grad

#import torch.nn as nn

# class Gaussian_Kernel_1D(object):
#     def __init__(self, sigma):
#         super().__init__()
#         self.sigma_ = nn.Parameter(torch.tensor(self.softplus_inv(torch.tensor(sigma))))
#
#     @property
#     def sigma(self):
#         return self.softplus(self.sigma_)
#
#     def softplus(self, x):
#         return torch.log(1 + torch.exp(x))
#
#     def softplus_inv(self, x):
#         return torch.log(torch.exp(x) - 1)
#
#     def kappa(self, x, y):
#         return torch.exp(-(1 / (2 * self.sigma ** 2)) * ((x - y) ** 2))
#
#     def Delta_x_kappa(self, x, y):
#         val = grad(grad(self.kappa, 0), 0)(x, y)
#         return val
#
#     def Delta_x_Delta_y_kappa(self, x, y):
#         val = grad(grad(self.Delta_x_kappa, 1), 1)(x, y)
#         return val


class Anisotropic_Mild_Kernel(object):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def kappa(self, ts, ty):
        x1 = ts[0]
        y1 = ty[1]
        x = ts[1:]
        y = ty[1:]

        scale_t = self.params[0]
        r = ((x1 - y1) / scale_t) ** 2

        a = self.params[1:5]
        theta = self.params[5:]
        a1, a2, a3, a4 = a
        t1, t2, t3, t4, t5, t6 = theta

        d2 = jnp.sum((x - y) ** 2)
        d = jnp.sqrt(jnp.sum((x - y) ** 2) + 1e-40)
        xy = jnp.dot(x, y)
        k1 = a1 ** 2 * (xy + t1 ** 2)
        k2 = a2 ** 2 * (t2**2 * xy + t3 ** 2) #** t4 # (jnp.abs(t4))
        k3 = a3 ** 2 * jnp.exp(-d2/(2 * t5 ** 2))
        k4 = a4 ** 2 * jnp.exp(-d/(2 * t6 ** 2))
        # k5 = a5 ** 2 * jnp.exp(-jnp.sin(jnp.pi * d2 / t7) ** 2 / t8 ** 2) * jnp.exp(-d2/t9 ** 2)
        # k6 = a6 ** 2 * jnp.exp(-jnp.sin(jnp.pi * d2 / t10) ** 2 / t11 ** 2)
        # k7 = a7 ** 2 * jnp.exp(-jnp.sin(jnp.pi * d / t12) ** 2 / t13 ** 2) * jnp.exp(-d / t14 ** 2)
        # k8 = a8 ** 2 * jnp.exp(-jnp.sin(jnp.pi * d/t15) ** 2 / t16 ** 2)
        # k9 = a9 ** 2 * (d2 + t17 ** 2) ** (1/2)
        # k10 = a10 ** 2 * (t18 ** 2 + t19 * d2) ** (-1/2)
        # k11 = a11 ** 2 * (t20 ** 2 + t21 ** 2 * d) ** (-1/2)
        # k12 = a12 ** 2 * (t22 ** 2 + d) ** t23
        # k13 = a13 ** 2 * (t24 ** 2 + d2) ** t25
        # k14 = a14 ** 2 * (1 + d2 / t26 ** 2) ** (-1)
        # k15 = a15 ** 2 * (1 + d/t27 ** 2) ** (-1)
        # k16 = a16 ** 2 * (1 - d2 / (d2 + t28 ** 2))
        # k17 = a17 ** 2 * jnp.maximum(0, 1 - d2/t29 ** 2)
        # k18 = a18 ** 2 * jnp.maximum(0, 1 - d/t30 ** 2)
        # k19 = a19 ** 2 * (jnp.log(d ** t31 + 1))
        # k20 = a20 ** 2 * jnp.tanh(t32 * xy + t33)
        # k21 = a21 ** 2 * (jnp.arccos(-d/t34 ** 2) - d/t34 ** 2 * jnp.sqrt(1 - d2 / t34 ** 4)) * (d2 < t34 ** 2)
        # return k1 + k2 + k3 + k4 + k5 + k6 + k7 + k8 + k9 + k10 + k11 \
        #     + k12 + k13 + k14 + k15 + k16 + k17 + k18 + k19 + k20 + k21
        return jnp.exp(-r) * (k1 + k2 + k3 + k4)



# class Good_Kernel(object):
#     def __init__(self, params):
#         super().__init__()
#         self.params = params
#
#     def kappa(self, x, y):
#         a = self.params[:2]
#         theta = self.params[2:]
#         a1, a2 = a
#         t1, t2 = theta
#
#         d2 = jnp.sum((x - y) ** 2)
#         d = jnp.sqrt(jnp.sum((x - y) ** 2) + 1e-20)
#         k1 = a1 ** 2 * jnp.exp(-d2/(2 * t1 ** 2))
#         k2 = a2 ** 2 * jnp.exp(-d/(2 * t2 ** 2))
#         return k1 + k2


# class Good_Kernel(object):
#     def __init__(self, params):
#         super().__init__()
#         self.params = params
#
#     def kappa(self, x, y):
#         # a = self.params[0]
#         theta = self.params
#         # a1 = a
#         t1 = theta
#
#         d2 = jnp.sum((x - y) ** 2)
#         d = jnp.sqrt(jnp.sum((x - y) ** 2) + 1e-20)
#         k1 = jnp.exp(-d2/(2 * t1 ** 2))
#         return k1


class Good_Kernel(object):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def kappa(self, x, y):
        a = self.params[:2]
        theta = self.params[2:]
        a3, a4 = a
        t5, t6 = theta

        d2 = jnp.sum((x - y) ** 2)
        d = jnp.sqrt(jnp.sum((x - y) ** 2) + 1e-40)
        k3 = a3 ** 2 * jnp.exp(-d2/(2 * t5 ** 2))
        k4 = a4 ** 2 * jnp.exp(-d/(2 * t6 ** 2))
        return k3 + k4

class Multi_Scale_Kernel(object):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def kappa(self, x, y):
        x1, x2 = x
        y1, y2 = y

        a = self.params[:3]
        theta = self.params[3:5]
        a1, a2, a3 = a
        t1, t2 = theta

        e1, e2, e3, e4, e5 = self.params[5:]

        d2 = jnp.sum((x - y) ** 2)
        d = jnp.sqrt(jnp.sum((x - y) ** 2) + 1e-40)

        k1 = a1 ** 2 * jnp.exp(-d2 / (2 * t1 ** 2))
        k2 = a2 ** 2 * jnp.exp(-d / (2 * t2 ** 2))
        k3 = a3 ** 2 * self.A(x1, x2, e1, e2, e3, e4, e5) * self.A(y1, y2, e1, e2, e3, e4, e5)
        return k1 + k2 + k3

    def A(self, x1, x2, e1, e2, e3, e4, e5):
        p1 = (1.1 + jnp.sin(2 * jnp.pi * x1 / e1)) / (1.1 + jnp.sin(2 * jnp.pi * x2 / e1))
        p2 = (1.1 + jnp.sin(2 * jnp.pi * x2 / e2)) / (1.1 + jnp.cos(2 * jnp.pi * x1 / e2))
        p3 = (1.1 + jnp.cos(2 * jnp.pi * x1 / e3)) / (1.1 + jnp.sin(2 * jnp.pi * x2 / e3))
        p4 = (1.1 + jnp.cos(2 * jnp.pi * x1 / e4)) / (1.1 + jnp.cos(2 * jnp.pi * x1 / e4))
        p5 = (1.1 + jnp.cos(2 * jnp.pi * x1 / e5)) / (1.1 + jnp.sin(2 * jnp.pi * x2 / e5))
        p6 = 1 + jnp.sin(4 * x1 ** 2 * x2 ** 2)
        return jnp.log((p1 + p2 + p3 + p4 + p5 + p6) / 6)

class Mild_Kernel(object):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def kappa(self, x, y):
        a = self.params[:4]
        theta = self.params[4:]
        a1, a2, a3, a4 = a
        t1, t2, t3, t4, t5, t6 = theta

        d2 = jnp.sum((x - y) ** 2)
        d = jnp.sqrt(jnp.sum((x - y) ** 2) + 1e-20)
        xy = jnp.dot(x, y)
        k1 = a1 ** 2 * (xy + t1 ** 2)
        k2 = a2 ** 2 * (t2**2 * xy + t3 ** 2) #** t4 # (jnp.abs(t4))
        k3 = a3 ** 2 * jnp.exp(-d2/(2 * t5 ** 2))
        k4 = a4 ** 2 * jnp.exp(-d/(2 * t6 ** 2))
        # k5 = a5 ** 2 * jnp.exp(-jnp.sin(jnp.pi * d2 / t7) ** 2 / t8 ** 2) * jnp.exp(-d2/t9 ** 2)
        # k6 = a6 ** 2 * jnp.exp(-jnp.sin(jnp.pi * d2 / t10) ** 2 / t11 ** 2)
        # k7 = a7 ** 2 * jnp.exp(-jnp.sin(jnp.pi * d / t12) ** 2 / t13 ** 2) * jnp.exp(-d / t14 ** 2)
        # k8 = a8 ** 2 * jnp.exp(-jnp.sin(jnp.pi * d/t15) ** 2 / t16 ** 2)
        # k9 = a9 ** 2 * (d2 + t17 ** 2) ** (1/2)
        # k10 = a10 ** 2 * (t18 ** 2 + t19 * d2) ** (-1/2)
        # k11 = a11 ** 2 * (t20 ** 2 + t21 ** 2 * d) ** (-1/2)
        # k12 = a12 ** 2 * (t22 ** 2 + d) ** t23
        # k13 = a13 ** 2 * (t24 ** 2 + d2) ** t25
        # k14 = a14 ** 2 * (1 + d2 / t26 ** 2) ** (-1)
        # k15 = a15 ** 2 * (1 + d/t27 ** 2) ** (-1)
        # k16 = a16 ** 2 * (1 - d2 / (d2 + t28 ** 2))
        # k17 = a17 ** 2 * jnp.maximum(0, 1 - d2/t29 ** 2)
        # k18 = a18 ** 2 * jnp.maximum(0, 1 - d/t30 ** 2)
        # k19 = a19 ** 2 * (jnp.log(d ** t31 + 1))
        # k20 = a20 ** 2 * jnp.tanh(t32 * xy + t33)
        # k21 = a21 ** 2 * (jnp.arccos(-d/t34 ** 2) - d/t34 ** 2 * jnp.sqrt(1 - d2 / t34 ** 4)) * (d2 < t34 ** 2)
        # return k1 + k2 + k3 + k4 + k5 + k6 + k7 + k8 + k9 + k10 + k11 \
        #     + k12 + k13 + k14 + k15 + k16 + k17 + k18 + k19 + k20 + k21
        return k1 + k2 + k3 + k4


class Super_Kernel(object):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def kappa(self, x, y):
        a = self.params[:21]
        theta = self.params[21:]
        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21 = a
        t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34 = theta

        d2 = jnp.sum((x - y) ** 2)
        d = jnp.sqrt(jnp.sum((x - y) ** 2) + 1e-40)
        xy = jnp.dot(x, y)
        k1 = a1 ** 2 * (xy + t1 ** 2)
        k2 = a2 ** 2 * (t2**2 * xy + t3 ** 2) ** (jnp.abs(t4))
        k3 = a3 ** 2 * jnp.exp(-d2/(2 * t5 ** 2))
        k4 = a4 ** 2 * jnp.exp(-d/(2 * t6 ** 2))
        k5 = a5 ** 2 * jnp.exp(-jnp.sin(jnp.pi * d2 / t7) ** 2 / t8 ** 2) * jnp.exp(-d2/t9 ** 2)
        k6 = a6 ** 2 * jnp.exp(-jnp.sin(jnp.pi * d2 / t10) ** 2 / t11 ** 2)
        k7 = a7 ** 2 * jnp.exp(-jnp.sin(jnp.pi * d / t12) ** 2 / t13 ** 2) * jnp.exp(-d / t14 ** 2)
        k8 = a8 ** 2 * jnp.exp(-jnp.sin(jnp.pi * d/t15) ** 2 / t16 ** 2)
        k9 = a9 ** 2 * (d2 + t17 ** 2) ** (1/2)
        k10 = a10 ** 2 * (t18 ** 2 + t19 * d2) ** (-1/2)
        k11 = a11 ** 2 * (t20 ** 2 + t21 ** 2 * d) ** (-1/2)
        k12 = a12 ** 2 * (t22 ** 2 + d) ** t23
        k13 = a13 ** 2 * (t24 ** 2 + d2) ** t25
        k14 = a14 ** 2 * (1 + d2 / t26 ** 2) ** (-1)
        k15 = a15 ** 2 * (1 + d/t27 ** 2) ** (-1)
        k16 = a16 ** 2 * (1 - d2 / (d2 + t28 ** 2))
        k17 = a17 ** 2 * jnp.maximum(0, 1 - d2/t29 ** 2)
        k18 = a18 ** 2 * jnp.maximum(0, 1 - d/t30 ** 2)
        k19 = a19 ** 2 * (jnp.log(d ** t31 + 1))
        k20 = a20 ** 2 * jnp.tanh(t32 * xy + t33)
        k21 = a21 ** 2 * (jnp.arccos(-d/t34 ** 2) - d/t34 ** 2 * jnp.sqrt(1 - d2 / t34 ** 4)) * (d2 < t34 ** 2)
        return k1 + k2 + k3 + k4 + k5 + k6 + k7 + k8 + k9 + k10 + k11 \
            + k12 + k13 + k14 + k15 + k16 + k17 + k18 + k19 + k20 + k21

class Kernel_Wrapper2D(object):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel

    def kappa(self, x1, x2, y1, y2):
        x = jnp.array([x1, x2])
        y = jnp.array([y1, y2])
        return self.kernel.kappa(x, y)

    def D_x1_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 0)(x1, x2, y1, y2)
        return val

    def D_x2_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 1)(x1, x2, y1, y2)
        return val

    def DD_x2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 2)(x1, x2, y1, y2)
        return val

    def D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 3)(x1, x2, y1, y2)
        return val

    def DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    def D_x1_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 2)(x1, x2, y1, y2)
        return val

    def D_x1_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 3)(x1, x2, y1, y2)
        return val

    def D_x1_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 0), 3), 3)(x1, x2, y1, y2)
        return val

    def D_x2_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 1), 3)(x1, x2, y1, y2)
        return val

    def D_x2_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.D_x2_kappa, 2)(x1, x2, y1, y2)
        return val

    def D_x2_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 1), 3), 3)(x1, x2, y1, y2)
        return val

    def DD_x2_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(grad(self.kappa, 1), 1), 3), 3)(x1, x2, y1, y2)
        return val

    def Delta_x_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 2), 2)(x1, x2, y1, y2)
        val += grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    def D_x1_Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_y_kappa, 0)(x1, x2, y1, y2)
        return val

    def D_x2_Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_y_kappa, 1)(x1, x2, y1, y2)
        return val

    def Delta_x_Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.Delta_y_kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.Delta_y_kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa, 2)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa, 3)(x1, x2, y1, y2)
        return val

class Gaussian_Kernel_1D(object):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def kappa(self, x, y):
        return jnp.exp(-(1 / (2 * self.sigma ** 2)) * ((x - y) ** 2))

    def D_x_kappa(self, x, y):
        val = grad(self.kappa, 0)(x, y)
        return val

    def D_y_kappa(self, x, y):
        val = grad(self.kappa, 1)(x, y)
        return val
    def D_x_D_y_kappa(self, x, y):
        val = grad(grad(self.kappa, 0), 1)(x, y)
        return val

    def D_x_DD_y_kappa(self, x, y):
        val = grad(grad(grad(self.kappa, 0), 1), 1)(x, y)
        return val

    def DD_x_D_y_kappa(self, x, y):
        val = grad(grad(grad(self.kappa, 0), 0), 1)(x, y)
        return val

    def DD_x_DD_y_kappa(self, x, y):
        val = grad(grad(grad(grad(self.kappa, 0), 0), 1), 1)(x, y)
        return val

    def DD_x_kappa(self, x, y):
        val = grad(grad(self.kappa, 0), 0)(x, y)
        return val

    def DD_y_kappa(self, x, y):
        val = grad(grad(self.kappa, 1), 1)(x, y)
        return val



class Matern5_2(object):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma
        return (1 + jnp.sqrt(5) * r / rho + 5 * r ** 2 / (3 * rho ** 2)) * jnp.exp(-jnp.sqrt(5) * r / rho)

    def D_x1_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma
        dK_r = (-5 / (3 * rho ** 2) - 5 * jnp.sqrt(5) * r / (3 * rho ** 3)) * jnp.exp(-jnp.sqrt(5) * r / rho) # dK/r
        return dK_r * (x1 - y1)

    def D_x2_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma
        dK_r = (-5 / (3 * rho ** 2) - 5 * jnp.sqrt(5) * r / (3 * rho ** 3)) * jnp.exp(-jnp.sqrt(5) * r / rho) # dK/r
        return dK_r * (x2 - y2)

    def D_y1_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma
        dK_r = (-5 / (3 * rho ** 2) - 5 * jnp.sqrt(5) * r / (3 * rho ** 3)) * jnp.exp(-jnp.sqrt(5) * r / rho) # dK/r
        return dK_r * (y1 - x1)

    def D_y2_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma
        dK_r = (-5 / (3 * rho ** 2) - 5 * jnp.sqrt(5) * r / (3 * rho ** 3)) * jnp.exp(-jnp.sqrt(5) * r / rho) # dK/r
        return dK_r * (y2 - x2)

    def D_x1_D_y1_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma
        return (5 / (3 * rho ** 2) + 5 * jnp.sqrt(5) * r / (3 * rho ** 3) - 25 * (x1 - y1) ** 2 / (3 * rho ** 4)) * jnp.exp(-jnp.sqrt(5) * r / rho)

    def D_x2_D_y2_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma
        return (5 / (3 * rho ** 2) + 5 * jnp.sqrt(5) * r / (3 * rho ** 3) - 25 * (x2 - y2) ** 2 / (3 * rho ** 4)) * jnp.exp(-jnp.sqrt(5) * r / rho)

    def D_x1_D_y2_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma
        return 25 * (x1 - y1) * (y2 - x2) / (3 * rho ** 4) * jnp.exp(-jnp.sqrt(5) * r / rho)

    def D_x2_D_y1_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma
        return 25 * (x2 - y2) * (y1 - x1) / (3 * rho ** 4) * jnp.exp(-jnp.sqrt(5) * r / rho)

    def Delta_x_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma
        return (-10 / (3 * rho ** 2) - 10 * jnp.sqrt(5) * r / (3 * rho ** 3)
                + 25 * r ** 2 / (3 * rho ** 4)) * jnp.exp(-jnp.sqrt(5) * r / rho)

    def Delta_y_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma
        return (-10 / (3 * rho ** 2) - 10 * jnp.sqrt(5) * r / (3 * rho ** 3)
                + 25 * r ** 2 / (3 * rho ** 4)) * jnp.exp(-jnp.sqrt(5) * r / rho)

    def D_x1_Delta_y_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma
        return (100 / (3 * rho ** 4) - 25 * jnp.sqrt(5) * r / (3 * rho ** 5)) * jnp.exp(-jnp.sqrt(5) * r / rho) * (
                    x1 - y1)

    def D_x2_Delta_y_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma
        return (100 / (3 * rho ** 4) - 25 * jnp.sqrt(5) * r / (3 * rho ** 5)) * jnp.exp(-jnp.sqrt(5) * r / rho) * (
                    x2 - y2)

    def Delta_x_D_y1_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma
        return (100 / (3 * rho ** 4) - 25 * jnp.sqrt(5) * r / (3 * rho ** 5)) * jnp.exp(-jnp.sqrt(5) * r / rho) * (y1 - x1)

    def Delta_x_D_y2_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma
        return (100 / (3 * rho ** 4) - 25 * jnp.sqrt(5) * r / (3 * rho ** 5)) * jnp.exp(-jnp.sqrt(5) * r / rho) * (y2 - x2)

    def Delta_x_Delta_y_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma
        return (200 / (3 * rho ** 4) - 175 * jnp.sqrt(5) * r / (3 * rho ** 5) + 125 * r ** 2/ (3 * rho ** 6)) * jnp.exp(-jnp.sqrt(5) * r / rho)

    def D_x1_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.kappa, 0)(x1, x2, y1, y2)
        return val

    def D_x1_D_y1_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.D_x1_kappa_jax, 2)(x1, x2, y1, y2)
        return val

    def D_x1_D_y2_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.D_x1_kappa_jax, 3)(x1, x2, y1, y2)
        return val

    def D_x2_D_y1_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.D_x2_kappa_jax, 2)(x1, x2, y1, y2)
        return val

    def D_x2_D_y2_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.D_x2_kappa_jax, 3)(x1, x2, y1, y2)
        return val

    def D_x2_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.kappa, 1)(x1, x2, y1, y2)
        return val

    def D_y1_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.kappa, 2)(x1, x2, y1, y2)
        return val

    def D_y2_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.kappa, 3)(x1, x2, y1, y2)
        return val

    def Delta_x_kappa_jax(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 0)(x1, x2, y1, y2)
        val = val + grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_y_kappa_jax(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 2), 2)(x1, x2, y1, y2)
        val = val + grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y1_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa_jax, 2)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y2_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa_jax, 3)(x1, x2, y1, y2)
        return val

    def D_x1_Delta_y_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.Delta_y_kappa_jax, 0)(x1, x2, y1, y2)
        return val

    def D_x2_Delta_y_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.Delta_y_kappa_jax, 1)(x1, x2, y1, y2)
        return val

    def Delta_x_Delta_y_kappa_jax(self, x1, x2, y1, y2):
        val = grad(grad(self.Delta_y_kappa_jax, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.Delta_y_kappa_jax, 1), 1)(x1, x2, y1, y2)
        return val


class Matern7_2(object):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        numerator = (7 * jnp.sqrt(7) * r ** 3 + 42 * r ** 2 * rho + 15 * jnp.sqrt(7) * r * rho ** 2 + 15 * rho ** 3)
        denominator = 15 * rho ** 3
        result = numerator * jnp.exp(-jnp.sqrt(7) * r / rho) / denominator
        return result

    def D_x1_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        numerator = -7 * jnp.exp(-jnp.sqrt(7) * r / rho) * (3 * rho ** 2 + 7 * r ** 2 + 3 * jnp.sqrt(7) * rho * r)
        denominator = 15 * rho ** 4
        result = numerator / denominator * (x1 - y1)
        return result

    def D_x2_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        numerator = -7 * jnp.exp(-jnp.sqrt(7) * r / rho) * (3 * rho ** 2 + 7 * r ** 2 + 3 * jnp.sqrt(7) * rho * r)
        denominator = 15 * rho ** 4
        result = numerator / denominator * (x2 - y2)
        return result

    def D_y1_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        numerator = -7 * jnp.exp(-jnp.sqrt(7) * r / rho) * (3 * rho ** 2 + 7 * r ** 2 + 3 * jnp.sqrt(7) * rho * r)
        denominator = 15 * rho ** 4
        result = numerator / denominator * (y1 - x1)
        return result

    def D_y2_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        numerator = -7 * jnp.exp(-jnp.sqrt(7) * r / rho) * (3 * rho ** 2 + 7 * r ** 2 + 3 * jnp.sqrt(7) * rho * r)
        denominator = 15 * rho ** 4
        result = numerator / denominator * (y2 - x2)
        return result

    def D_x1_D_y1_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        # K'r^{-1}
        dK_r = (-7 * jnp.exp(-jnp.sqrt(7) * r / rho) * (3 * rho ** 2 + 7 * r ** 2 + 3 * jnp.sqrt(7) * rho * r)) / (15 * rho ** 4)

        numerator = -49 * jnp.exp(-jnp.sqrt(7) * r / rho) * (rho + jnp.sqrt(7) * r)
        denominator = 15 * rho ** 5
        result = numerator / denominator * (x1 - y1) ** 2 - dK_r
        return  result
    def D_x2_D_y2_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        # K'r^{-1}
        dK_r = (-7 * jnp.exp(-jnp.sqrt(7) * r / rho) * (3 * rho ** 2 + 7 * r ** 2 + 3 * jnp.sqrt(7) * rho * r)) / (
                    15 * rho ** 4)

        numerator = -49 * jnp.exp(-jnp.sqrt(7) * r / rho) * (rho + jnp.sqrt(7) * r)
        denominator = 15 * rho ** 5
        result = numerator / denominator * (x2 - y2) ** 2 - dK_r
        return result

    def D_x1_D_y2_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        numerator = -49 * jnp.exp(-jnp.sqrt(7) * r / rho) * (rho + jnp.sqrt(7) * r)
        denominator = 15 * rho ** 5
        result = numerator / denominator * (x1 - y1) * (x2 - y2)
        return result

    def D_x2_D_y1_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        numerator = -49 * jnp.exp(-jnp.sqrt(7) * r / rho) * (rho + jnp.sqrt(7) * r)
        denominator = 15 * rho ** 5
        result = numerator / denominator * (x2 - y2) * (x1 - y1)
        return result

    def Delta_x_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        numerator = -7 * jnp.exp(-jnp.sqrt(7) * r / rho) * (
                    6 * rho ** 3 - 7 * jnp.sqrt(7) * r ** 3 + 7 * rho * r ** 2 + 6 * jnp.sqrt(7) * rho ** 2 * r)
        denominator = 15 * rho ** 5
        result = numerator / denominator
        return result

    def Delta_y_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        numerator = -7 * jnp.exp(-jnp.sqrt(7) * r / rho) * (
                6 * rho ** 3 - 7 * jnp.sqrt(7) * r ** 3 + 7 * rho * r ** 2 + 6 * jnp.sqrt(7) * rho ** 2 * r)
        denominator = 15 * rho ** 5
        result = numerator / denominator
        return result

    def D_x1_Delta_y_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        numerator = 49 * jnp.exp(-jnp.sqrt(7) * r / rho) * (4 * rho ** 2 - 7 * r ** 2 + 4 * jnp.sqrt(7) * rho * r)
        denominator = 15 * rho ** 6
        result = numerator / denominator * (x1 - y1)
        return result


    def D_x2_Delta_y_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        numerator = 49 * jnp.exp(-jnp.sqrt(7) * r / rho) * (4 * rho ** 2 - 7 * r ** 2 + 4 * jnp.sqrt(7) * rho * r)
        denominator = 15 * rho ** 6
        result = numerator / denominator * (x2 - y2)
        return result

    def Delta_x_D_y1_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        numerator = 49 * jnp.exp(-jnp.sqrt(7) * r / rho) * (4 * rho ** 2 - 7 * r ** 2 + 4 * jnp.sqrt(7) * rho * r)
        denominator = 15 * rho ** 6
        result = numerator / denominator * (y1 - x1)
        return result
    def Delta_x_D_y2_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        numerator = 49 * jnp.exp(-jnp.sqrt(7) * r / rho) * (4 * rho ** 2 - 7 * r ** 2 + 4 * jnp.sqrt(7) * rho * r)
        denominator = 15 * rho ** 6
        result = numerator / denominator * (y2 - x2)
        return result
    def Delta_x_Delta_y_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        numerator = 49 * jnp.exp(-jnp.sqrt(7) * r / rho) * (
                    8 * rho ** 3 + 7 * jnp.sqrt(7) * r ** 3 - 56 * rho * r ** 2 + 8 * jnp.sqrt(7) * rho ** 2 * r)
        denominator = 15 * rho ** 7
        result = numerator / denominator
        return result
    def D_x1_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.kappa, 0)(x1, x2, y1, y2)
        return val

    def D_x1_D_y1_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.D_x1_kappa_jax, 2)(x1, x2, y1, y2)
        return val

    def D_x1_D_y2_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.D_x1_kappa_jax, 3)(x1, x2, y1, y2)
        return val

    def D_x2_D_y1_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.D_x2_kappa_jax, 2)(x1, x2, y1, y2)
        return val

    def D_x2_D_y2_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.D_x2_kappa_jax, 3)(x1, x2, y1, y2)
        return val

    def D_x2_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.kappa, 1)(x1, x2, y1, y2)
        return val

    def D_y1_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.kappa, 2)(x1, x2, y1, y2)
        return val

    def D_y2_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.kappa, 3)(x1, x2, y1, y2)
        return val

    def Delta_x_kappa_jax(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 0)(x1, x2, y1, y2)
        val = val + grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_y_kappa_jax(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 2), 2)(x1, x2, y1, y2)
        val = val + grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y1_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa_jax, 2)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y2_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa_jax, 3)(x1, x2, y1, y2)
        return val

    def D_x1_Delta_y_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.Delta_y_kappa_jax, 0)(x1, x2, y1, y2)
        return val

    def D_x2_Delta_y_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.Delta_y_kappa_jax, 1)(x1, x2, y1, y2)
        return val

    def Delta_x_Delta_y_kappa_jax(self, x1, x2, y1, y2):
        val = grad(grad(self.Delta_y_kappa_jax, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.Delta_y_kappa_jax, 1), 1)(x1, x2, y1, y2)
        return val


class Matern9_2(object):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        numerator = (35 * rho ** 4 + 105 * rho ** 3 * r + 135 * rho ** 2 * r ** 2 + 90 * rho * r ** 3 + 27 * r ** 4)
        denominator = 35 * rho ** 4
        result = numerator * jnp.exp(-3 * r / rho) / denominator
        return result

    def D_x1_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        numerator = -9 * jnp.exp(-3 * r / rho) * (5 * rho ** 3 + 9 * r ** 3 + 18 * rho * r ** 2 + 15 * rho ** 2 * r)
        denominator = 35 * rho ** 5
        result = numerator / denominator * (x1 - y1)
        return result

    def D_x2_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        numerator = -9 * jnp.exp(-3 * r / rho) * (5 * rho ** 3 + 9 * r ** 3 + 18 * rho * r ** 2 + 15 * rho ** 2 * r)
        denominator = 35 * rho ** 5
        result = numerator / denominator * (x2 - y2)
        return result

    def D_y1_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        numerator = -9 * jnp.exp(-3 * r / rho) * (5 * rho ** 3 + 9 * r ** 3 + 18 * rho * r ** 2 + 15 * rho ** 2 * r)
        denominator = 35 * rho ** 5
        result = numerator / denominator * (y1 - x1)
        return result

    def D_y2_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        numerator = -9 * jnp.exp(-3 * r / rho) * (5 * rho ** 3 + 9 * r ** 3 + 18 * rho * r ** 2 + 15 * rho ** 2 * r)
        denominator = 35 * rho ** 5
        result = numerator / denominator * (y2 - x2)
        return result

    def D_x1_D_y1_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        # K'r^{-1}
        dK_r = (-9 * jnp.exp(-3 * r / rho) * (5 * rho ** 3 + 9 * r ** 3 + 18 * rho * r ** 2 + 15 * rho ** 2 * r))/(35 * rho ** 5)

        numerator = -81 * jnp.exp(-3 * r / rho) * (rho ** 2 + 3 * r ** 2 + 3 * rho * r)
        denominator = 35 * rho ** 6
        result = numerator / denominator * (x1 - y1) ** 2- dK_r
        return result
    def D_x2_D_y2_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        # K'r^{-1}
        dK_r = (-9 * jnp.exp(-3 * r / rho) * (5 * rho ** 3 + 9 * r ** 3 + 18 * rho * r ** 2 + 15 * rho ** 2 * r)) / (
                    35 * rho ** 5)

        numerator = -81 * jnp.exp(-3 * r / rho) * (rho ** 2 + 3 * r ** 2 + 3 * rho * r)
        denominator = 35 * rho ** 6
        result = numerator / denominator * (x2 - y2) ** 2 - dK_r
        return result

    def D_x1_D_y2_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        numerator = -81 * jnp.exp(-3 * r / rho) * (rho ** 2 + 3 * r ** 2 + 3 * rho * r)
        denominator = 35 * rho ** 6
        result = numerator / denominator * (x1 - y1) * (x2 - y2)
        return result

    def D_x2_D_y1_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        numerator = -81 * jnp.exp(-3 * r / rho) * (rho ** 2 + 3 * r ** 2 + 3 * rho * r)
        denominator = 35 * rho ** 6
        result = numerator / denominator * (x2 - y2) * (x1 - y1)
        return result

    def Delta_x_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        numerator = -9 * jnp.exp(-3 * r / rho) * (
                    10 * rho ** 4 - 27 * r ** 4 - 9 * rho * r ** 3 + 27 * rho ** 2 * r ** 2 + 30 * rho ** 3 * r)
        denominator = 35 * rho ** 6
        result = numerator / denominator
        return result

    def Delta_y_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        numerator = -9 * jnp.exp(-3 * r / rho) * (
                    10 * rho ** 4 - 27 * r ** 4 - 9 * rho * r ** 3 + 27 * rho ** 2 * r ** 2 + 30 * rho ** 3 * r)
        denominator = 35 * rho ** 6
        result = numerator / denominator
        return result

    def D_x1_Delta_y_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        numerator = 81 * jnp.exp(-3 * r / rho) * (4 * rho ** 3 - 9 * r ** 3 + 9 * rho * r ** 2 + 12 * rho ** 2 * r)
        denominator = 35 * rho ** 7
        result = numerator / denominator * (x1 - y1)
        return result


    def D_x2_Delta_y_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        numerator = 81 * jnp.exp(-3 * r / rho) * (4 * rho ** 3 - 9 * r ** 3 + 9 * rho * r ** 2 + 12 * rho ** 2 * r)
        denominator = 35 * rho ** 7
        result = numerator / denominator * (x2 - y2)
        return result

    def Delta_x_D_y1_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        numerator = 81 * jnp.exp(-3 * r / rho) * (4 * rho ** 3 - 9 * r ** 3 + 9 * rho * r ** 2 + 12 * rho ** 2 * r)
        denominator = 35 * rho ** 7
        result = numerator / denominator * (y1 - x1)
        return result
    def Delta_x_D_y2_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        numerator = 81 * jnp.exp(-3 * r / rho) * (4 * rho ** 3 - 9 * r ** 3 + 9 * rho * r ** 2 + 12 * rho ** 2 * r)
        denominator = 35 * rho ** 7
        result = numerator / denominator * (y2 - x2)
        return result
    def Delta_x_Delta_y_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        numerator = 81 * jnp.exp(-3 * r / rho) * (8 * rho ** 4 + 27 * r ** 4 - 72 * rho * r ** 3 + 24 * rho ** 3 * r)
        denominator = 35 * rho ** 8
        result = numerator / denominator
        return result
    def D_x1_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.kappa, 0)(x1, x2, y1, y2)
        return val

    def D_x1_D_y1_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.D_x1_kappa_jax, 2)(x1, x2, y1, y2)
        return val

    def D_x1_D_y2_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.D_x1_kappa_jax, 3)(x1, x2, y1, y2)
        return val

    def D_x2_D_y1_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.D_x2_kappa_jax, 2)(x1, x2, y1, y2)
        return val

    def D_x2_D_y2_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.D_x2_kappa_jax, 3)(x1, x2, y1, y2)
        return val

    def D_x2_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.kappa, 1)(x1, x2, y1, y2)
        return val

    def D_y1_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.kappa, 2)(x1, x2, y1, y2)
        return val

    def D_y2_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.kappa, 3)(x1, x2, y1, y2)
        return val

    def Delta_x_kappa_jax(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 0)(x1, x2, y1, y2)
        val = val + grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_y_kappa_jax(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 2), 2)(x1, x2, y1, y2)
        val = val + grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y1_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa_jax, 2)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y2_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa_jax, 3)(x1, x2, y1, y2)
        return val

    def D_x1_Delta_y_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.Delta_y_kappa_jax, 0)(x1, x2, y1, y2)
        return val

    def D_x2_Delta_y_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.Delta_y_kappa_jax, 1)(x1, x2, y1, y2)
        return val

    def Delta_x_Delta_y_kappa_jax(self, x1, x2, y1, y2):
        val = grad(grad(self.Delta_y_kappa_jax, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.Delta_y_kappa_jax, 1), 1)(x1, x2, y1, y2)
        return val

class Gaussian(object):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        exponent = -r ** 2 / (2 * rho ** 2)
        result = jnp.exp(exponent)
        return result

    def D_x1_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        exponent = -r ** 2 / (2 * rho ** 2)
        numerator = -jnp.exp(exponent)
        denominator = rho ** 2
        result = numerator / denominator * (x1 - y1)
        return result

    def D_x2_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        exponent = -r ** 2 / (2 * rho ** 2)
        numerator = -jnp.exp(exponent)
        denominator = rho ** 2
        result = numerator / denominator * (x2 - y2)
        return result

    def D_y1_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        exponent = -r ** 2 / (2 * rho ** 2)
        numerator = -jnp.exp(exponent)
        denominator = rho ** 2
        result = numerator / denominator * (y1 - x1)
        return result

    def D_y2_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        exponent = -r ** 2 / (2 * rho ** 2)
        numerator = -jnp.exp(exponent)
        denominator = rho ** 2
        result = numerator / denominator * (y2 - x2)
        return result

    def D_x1_D_y1_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        # K'r^{-1}
        dK_r = -jnp.exp(-r ** 2 / (2 * rho ** 2)) / (rho ** 2)

        exponent = -r ** 2 / (2 * rho ** 2)
        numerator = -jnp.exp(exponent)
        denominator = rho ** 4
        result = numerator / denominator * (x1 - y1) ** 2 - dK_r
        return result
    def D_x2_D_y2_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        # K'r^{-1}
        dK_r = -jnp.exp(-r ** 2 / (2 * rho ** 2)) / (rho ** 2)

        exponent = -r ** 2 / (2 * rho ** 2)
        numerator = -jnp.exp(exponent)
        denominator = rho ** 4
        result = numerator / denominator * (x2 - y2) ** 2 - dK_r
        return result

    def D_x1_D_y2_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        exponent = -r ** 2 / (2 * rho ** 2)
        numerator = -jnp.exp(exponent)
        denominator = rho ** 4
        result = numerator / denominator * (x1 - y1) * (x2 - y2)
        return result

    def D_x2_D_y1_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        exponent = -r ** 2 / (2 * rho ** 2)
        numerator = -jnp.exp(exponent)
        denominator = rho ** 4
        result = numerator / denominator * (x1 - y1) * (x2 - y2)
        return result

    def Delta_x_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        exponent = -r ** 2 / (2 * rho ** 2)
        numerator = jnp.exp(exponent) * (r ** 2 - 2 * rho ** 2)
        denominator = rho ** 4
        result = numerator / denominator
        return result

    def Delta_y_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        exponent = -r ** 2 / (2 * rho ** 2)
        numerator = jnp.exp(exponent) * (r ** 2 - 2 * rho ** 2)
        denominator = rho ** 4
        result = numerator / denominator
        return result

    def D_x1_Delta_y_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        exponent = -r ** 2 / (2 * rho ** 2)
        numerator = jnp.exp(exponent) * (4 * rho ** 2 - r ** 2)
        denominator = rho ** 6
        result = numerator / denominator * (x1 - y1)
        return result

    def D_x2_Delta_y_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        exponent = -r ** 2 / (2 * rho ** 2)
        numerator = jnp.exp(exponent) * (4 * rho ** 2 - r ** 2)
        denominator = rho ** 6
        result = numerator / denominator * (x2 - y2)
        return result

    def Delta_x_D_y1_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        exponent = -r ** 2 / (2 * rho ** 2)
        numerator = jnp.exp(exponent) * (4 * rho ** 2 - r ** 2)
        denominator = rho ** 6
        result = numerator / denominator * (y1 - x1)
        return result
    def Delta_x_D_y2_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        exponent = -r ** 2 / (2 * rho ** 2)
        numerator = jnp.exp(exponent) * (4 * rho ** 2 - r ** 2)
        denominator = rho ** 6
        result = numerator / denominator * (y2 - x2)
        return result
    def Delta_x_Delta_y_kappa(self, x1, x2, y1, y2):
        r = jnp.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
        rho = self.sigma

        exponent = -r ** 2 / (2 * rho ** 2)
        numerator = jnp.exp(exponent) * (8 * rho ** 4 + r ** 4 - 8 * rho ** 2 * r ** 2)
        denominator = rho ** 8
        result = numerator / denominator
        return result
    def D_x1_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.kappa, 0)(x1, x2, y1, y2)
        return val

    def D_x1_D_y1_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.D_x1_kappa_jax, 2)(x1, x2, y1, y2)
        return val

    def D_x1_D_y2_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.D_x1_kappa_jax, 3)(x1, x2, y1, y2)
        return val

    def D_x2_D_y1_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.D_x2_kappa_jax, 2)(x1, x2, y1, y2)
        return val

    def D_x2_D_y2_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.D_x2_kappa_jax, 3)(x1, x2, y1, y2)
        return val

    def D_x2_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.kappa, 1)(x1, x2, y1, y2)
        return val

    def D_y1_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.kappa, 2)(x1, x2, y1, y2)
        return val

    def D_y2_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.kappa, 3)(x1, x2, y1, y2)
        return val

    def Delta_x_kappa_jax(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 0)(x1, x2, y1, y2)
        val = val + grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_y_kappa_jax(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 2), 2)(x1, x2, y1, y2)
        val = val + grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y1_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa_jax, 2)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y2_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa_jax, 3)(x1, x2, y1, y2)
        return val

    def D_x1_Delta_y_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.Delta_y_kappa_jax, 0)(x1, x2, y1, y2)
        return val

    def D_x2_Delta_y_kappa_jax(self, x1, x2, y1, y2):
        val = grad(self.Delta_y_kappa_jax, 1)(x1, x2, y1, y2)
        return val

    def Delta_x_Delta_y_kappa_jax(self, x1, x2, y1, y2):
        val = grad(grad(self.Delta_y_kappa_jax, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.Delta_y_kappa_jax, 1), 1)(x1, x2, y1, y2)
        return val



class Matern_Kernel(object):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def kappa(self, x1, x2, y1, y2):
        d = jnp.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        # return (1 + jnp.sqrt(3) * d / self.sigma) * jnp.exp(-jnp.sqrt(3) * d / self.sigma)
        return (1 + jnp.sqrt(5) * d / self.sigma + 5 * d**2 / (3 * self.sigma ** 2)) * jnp.exp(- jnp.sqrt(5) * d / self.sigma)
    def D_x1_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 0)(x1, x2, y1, y2)
        return val

    def D_x2_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 1)(x1, x2, y1, y2)
        return val

    def DD_x2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 2)(x1, x2, y1, y2)
        return val

    def D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 3)(x1, x2, y1, y2)
        return val

    def DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    def D_x1_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 2)(x1, x2, y1, y2)
        return val

    def D_x1_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 3)(x1, x2, y1, y2)
        return val

    def D_x1_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 0), 3), 3)(x1, x2, y1, y2)
        return val

    def D_x2_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 1), 3)(x1, x2, y1, y2)
        return val

    def D_x2_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.D_x2_kappa, 2)(x1, x2, y1, y2)
        return val

    def D_x2_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 1), 3), 3)(x1, x2, y1, y2)
        return val

    def DD_x2_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(grad(self.kappa, 1), 1), 3), 3)(x1, x2, y1, y2)
        return val

    def Delta_x_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 2), 2)(x1, x2, y1, y2)
        val += grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    def D_x1_Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_y_kappa, 0)(x1, x2, y1, y2)
        return val

    def D_x2_Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_y_kappa, 1)(x1, x2, y1, y2)
        return val

    def Delta_x_Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.Delta_y_kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.Delta_y_kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa, 2)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa, 3)(x1, x2, y1, y2)
        return val

class Gaussian_Kernel(object):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def kappa(self, x1, x2, y1, y2):
        return jnp.exp(-(1 / (2 * self.sigma ** 2)) * ((x1 - y1) ** 2 + (x2 - y2) ** 2))

    def D_x1_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 0)(x1, x2, y1, y2)
        return val

    def D_x2_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 1)(x1, x2, y1, y2)
        return val

    def DD_x2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 2)(x1, x2, y1, y2)
        return val

    def D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 3)(x1, x2, y1, y2)
        return val

    def DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    def D_x1_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 2)(x1, x2, y1, y2)
        return val

    def D_x1_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 3)(x1, x2, y1, y2)
        return val

    def D_x1_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 0), 3), 3)(x1, x2, y1, y2)
        return val

    def D_x2_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 1), 3)(x1, x2, y1, y2)
        return val

    def D_x2_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.D_x2_kappa, 2)(x1, x2, y1, y2)
        return val

    def D_x2_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 1), 3), 3)(x1, x2, y1, y2)
        return val

    def DD_x2_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(grad(self.kappa, 1), 1), 3), 3)(x1, x2, y1, y2)
        return val

    def Delta_x_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 2), 2)(x1, x2, y1, y2)
        val += grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    def D_x1_Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_y_kappa, 0)(x1, x2, y1, y2)
        return val

    def D_x2_Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_y_kappa, 1)(x1, x2, y1, y2)
        return val

    def Delta_x_Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.Delta_y_kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.Delta_y_kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa, 2)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa, 3)(x1, x2, y1, y2)
        return val

class Anisotropic_Gaussian_kernel(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def kappa(self, x1, x2, y1, y2):
        scale_t = self.sigma[0]
        scale_x = self.sigma[1]
        r = ((x1 - y1) / scale_t) ** 2 + ((x2 - y2) / scale_x) ** 2
        return jnp.exp(-r)

    def D_x1_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 0)(x1, x2, y1, y2)
        return val

    def D_x2_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 1)(x1, x2, y1, y2)
        return val

    def DD_x2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 2)(x1, x2, y1, y2)
        return val

    def D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 3)(x1, x2, y1, y2)
        return val

    def DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    def D_x1_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 2)(x1, x2, y1, y2)
        return val

    def D_x1_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 3)(x1, x2, y1, y2)
        return val

    def D_x1_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 0), 3), 3)(x1, x2, y1, y2)
        return val

    def D_x2_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 1), 3)(x1, x2, y1, y2)
        return val

    def D_x2_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.D_x2_kappa, 2)(x1, x2, y1, y2)
        return val

    def D_x2_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 1), 3), 3)(x1, x2, y1, y2)
        return val

    def DD_x2_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 1), 1), 2)(x1, x2, y1, y2)
        return val

    def DD_x2_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 1), 1), 3)(x1, x2, y1, y2)
        return val

    def DD_x2_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(grad(self.kappa, 1), 1), 3), 3)(x1, x2, y1, y2)
        return val

    def Delta_x_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 2), 2)(x1, x2, y1, y2)
        val += grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    def Delta_x_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.Delta_y_kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.Delta_y_kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_x_Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.Delta_y_kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.Delta_y_kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa, 2)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa, 3)(x1, x2, y1, y2)
        return val

    def D_x1_Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_y_kappa, 0)(x1, x2, y1, y2)
        return val

    def D_x2_Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_y_kappa, 1)(x1, x2, y1, y2)
        return val

class Anisotropic_Periodic_Gaussian_kernel(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def kappa(self, x1, x2, y1, y2):
        scale_t = self.sigma[0]
        scale_x = self.sigma[1]
        r1 = ((x1 - y1) / scale_t) ** 2
        r2 = jnp.cos(jnp.pi * (x2 - y2)) - 1
        return jnp.exp(-r1 + r2/(scale_x ** 2))

        # r2 = 2 * jnp.sin(jnp.pi*(x2-y2)/2)**2
        # return jnp.exp(-r1 - r2/(scale_x ** 2))


    def D_x1_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 0)(x1, x2, y1, y2)
        return val

    def D_x2_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 1)(x1, x2, y1, y2)
        return val

    def DD_x2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 2)(x1, x2, y1, y2)
        return val

    def D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 3)(x1, x2, y1, y2)
        return val

    def DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    def D_x1_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 2)(x1, x2, y1, y2)
        return val

    def D_x1_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 3)(x1, x2, y1, y2)
        return val

    def D_x1_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 0), 3), 3)(x1, x2, y1, y2)
        return val

    def D_x2_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 1), 3)(x1, x2, y1, y2)
        return val

    def D_x2_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.D_x2_kappa, 2)(x1, x2, y1, y2)
        return val

    def D_x2_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 1), 3), 3)(x1, x2, y1, y2)
        return val

    def DD_x2_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 1), 1), 2)(x1, x2, y1, y2)
        return val

    def DD_x2_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 1), 1), 3)(x1, x2, y1, y2)
        return val

    def DD_x2_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(grad(self.kappa, 1), 1), 3), 3)(x1, x2, y1, y2)
        return val

    def Delta_x_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 2), 2)(x1, x2, y1, y2)
        val += grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    def Delta_x_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.Delta_y_kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.Delta_y_kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_x_Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.Delta_y_kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.Delta_y_kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa, 2)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa, 3)(x1, x2, y1, y2)
        return val




class Anisotropic_Kernel_Wrapper1D(object):
    def __init__(self, kernel):
        self.kernel = kernel

    def kappa(self, x1, x2, y1, y2):
        x = jnp.array([x1, x2])
        y = jnp.array([y1, y2])
        return self.kernel.kappa(x, y)

    def D_x1_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 0)(x1, x2, y1, y2)
        return val

    def D_x2_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 1)(x1, x2, y1, y2)
        return val

    def DD_x2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 2)(x1, x2, y1, y2)
        return val

    def D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 3)(x1, x2, y1, y2)
        return val

    def DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    def D_x1_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 2)(x1, x2, y1, y2)
        return val

    def D_x1_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 3)(x1, x2, y1, y2)
        return val

    def D_x1_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 0), 3), 3)(x1, x2, y1, y2)
        return val

    def D_x2_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 1), 3)(x1, x2, y1, y2)
        return val

    def D_x2_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.D_x2_kappa, 2)(x1, x2, y1, y2)
        return val

    def D_x2_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 1), 3), 3)(x1, x2, y1, y2)
        return val

    def DD_x2_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 1), 1), 2)(x1, x2, y1, y2)
        return val

    def DD_x2_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 1), 1), 3)(x1, x2, y1, y2)
        return val

    def DD_x2_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(grad(self.kappa, 1), 1), 3), 3)(x1, x2, y1, y2)
        return val

    def Delta_x_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 2), 2)(x1, x2, y1, y2)
        val += grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    def Delta_x_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.Delta_y_kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.Delta_y_kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_x_Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.Delta_y_kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.Delta_y_kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa, 2)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa, 3)(x1, x2, y1, y2)
        return val


class Time_Dependent_Kernel_Wrapper1D(object):
    def __init__(self, kernel_t, kernel_s):
        self.kernel_t = kernel_t
        self.kernel_s = kernel_s

    def kappa(self, x1, x2, y1, y2):
        return self.kernel_t.kappa(x1, y1) * self.kernel_s.kappa(x2, y2)

    def D_x1_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 0)(x1, x2, y1, y2)
        return val

    def D_x2_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 1)(x1, x2, y1, y2)
        return val

    def DD_x2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 2)(x1, x2, y1, y2)
        return val

    def D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 3)(x1, x2, y1, y2)
        return val

    def DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    def D_x1_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 2)(x1, x2, y1, y2)
        return val

    def D_x1_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 3)(x1, x2, y1, y2)
        return val

    def D_x1_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 0), 3), 3)(x1, x2, y1, y2)
        return val

    def D_x2_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 1), 3)(x1, x2, y1, y2)
        return val

    def D_x2_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.D_x2_kappa, 2)(x1, x2, y1, y2)
        return val

    def D_x2_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 1), 3), 3)(x1, x2, y1, y2)
        return val

    def DD_x2_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 1), 1), 2)(x1, x2, y1, y2)
        return val

    def DD_x2_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 1), 1), 3)(x1, x2, y1, y2)
        return val

    def DD_x2_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(grad(self.kappa, 1), 1), 3), 3)(x1, x2, y1, y2)
        return val

    def Delta_x_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 2), 2)(x1, x2, y1, y2)
        val += grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    def Delta_x_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.Delta_y_kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.Delta_y_kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_x_Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.Delta_y_kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.Delta_y_kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa, 2)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa, 3)(x1, x2, y1, y2)
        return val



class Anisotropic_Matern52_kernel(object):
    def __init__(self, sigma_t, sigma_x, epsilon=1e-20):
        self.sigma_t = sigma_t
        self.sigma_x = sigma_x
        self.epsilon = epsilon

    def matern52(self, r):
        """Matérn 5/2 kernel function."""
        return (1 + jnp.sqrt(5) * r + (5/3) * r**2) * jnp.exp(-jnp.sqrt(5) * r)

    def compute_distance(self, x1, y1, scale):
        """Compute the distance with an added epsilon to prevent issues with auto-differentiation."""
        r_squared = (jnp.sum((x1 - y1) ** 2) + self.epsilon) * scale ** 2
        return jnp.sqrt(r_squared)

    def kappa(self, x1, x2, y1, y2):
        r_t = self.compute_distance(x1, y1, self.sigma_t)
        r_x = self.compute_distance(x2, y2, self.sigma_x)

        matern_t = self.matern52(r_t)
        matern_x = self.matern52(r_x)

        return matern_t * matern_x

    def D_x1_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 0)(x1, x2, y1, y2)
        return val

    def D_x2_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 1)(x1, x2, y1, y2)
        return val

    def DD_x2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 2)(x1, x2, y1, y2)
        return val

    def D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 3)(x1, x2, y1, y2)
        return val

    def DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    def D_x1_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 2)(x1, x2, y1, y2)
        return val

    def D_x1_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 3)(x1, x2, y1, y2)
        return val

    def D_x1_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 0), 3), 3)(x1, x2, y1, y2)
        return val

    def D_x2_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 1), 3)(x1, x2, y1, y2)
        return val

    def D_x2_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.D_x2_kappa, 2)(x1, x2, y1, y2)
        return val

    def D_x2_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 1), 3), 3)(x1, x2, y1, y2)
        return val

    def DD_x2_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 1), 1), 2)(x1, x2, y1, y2)
        return val

    def DD_x2_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 1), 1), 3)(x1, x2, y1, y2)
        return val

    def DD_x2_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(grad(self.kappa, 1), 1), 3), 3)(x1, x2, y1, y2)
        return val

    def Delta_x_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 2), 2)(x1, x2, y1, y2)
        val += grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    def Delta_x_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.Delta_y_kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.Delta_y_kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_x_Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.Delta_y_kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.Delta_y_kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa, 2)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa, 3)(x1, x2, y1, y2)
        return val

class Anisotropic_Mixed_kernel(object):
    def __init__(self, sigma_t, sigma_x, epsilon=1e-6):
        self.sigma_t = sigma_t
        self.sigma_x = sigma_x
        self.epsilon = epsilon

    def matern52(self, r):
        """Matérn 5/2 kernel function."""
        return (1 + jnp.sqrt(5) * r + (5/3) * r**2) * jnp.exp(-jnp.sqrt(5) * r)

    def compute_distance(self, x1, y1, scale):
        """Compute the distance with an added epsilon to prevent issues with auto-differentiation."""
        r_squared = jnp.sum((x1 - y1) ** 2 + self.epsilon) / scale ** 2
        return jnp.sqrt(r_squared)

    def kappa(self, x1, x2, y1, y2):
        # Gaussian kernel for the time component
        d_t = ((x1 - y1) / self.sigma_t) ** 2
        gaussian_t = jnp.exp(-d_t)

        # Matérn 5/2 kernel for the space component
        r_x = self.compute_distance(x2, y2, self.sigma_x)
        matern_x = self.matern52(r_x)

        return gaussian_t * matern_x

    def D_x1_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 0)(x1, x2, y1, y2)
        return val

    def D_x2_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 1)(x1, x2, y1, y2)
        return val

    def DD_x2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 2)(x1, x2, y1, y2)
        return val

    def D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 3)(x1, x2, y1, y2)
        return val

    def DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    def D_x1_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 2)(x1, x2, y1, y2)
        return val

    def D_x1_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 3)(x1, x2, y1, y2)
        return val

    def D_x1_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 0), 3), 3)(x1, x2, y1, y2)
        return val

    def D_x2_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 1), 3)(x1, x2, y1, y2)
        return val

    def D_x2_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.D_x2_kappa, 2)(x1, x2, y1, y2)
        return val

    def D_x2_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 1), 3), 3)(x1, x2, y1, y2)
        return val

    def DD_x2_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 1), 1), 2)(x1, x2, y1, y2)
        return val

    def DD_x2_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 1), 1), 3)(x1, x2, y1, y2)
        return val

    def DD_x2_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(grad(self.kappa, 1), 1), 3), 3)(x1, x2, y1, y2)
        return val

    def Delta_x_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 2), 2)(x1, x2, y1, y2)
        val += grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    def Delta_x_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.Delta_y_kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.Delta_y_kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_x_Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.Delta_y_kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.Delta_y_kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa, 2)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa, 3)(x1, x2, y1, y2)
        return val



class Periodic_Kernel_2D(object):
    def __init__(self, sigma, l):
        self.sigma = sigma  # variance
        self.l = l  # period

    def kappa(self, x1, x2, y1, y2):
        r1 = 1/(self.sigma) * (jnp.cos(2 * jnp.pi * (x1 - y1)) - 1)
        r2 = 1/(self.sigma) * (jnp.cos(2 * jnp.pi * (x2 - y2)) - 1)
        return jnp.exp(r1 + r2)
        #r = jnp.sin(jnp.pi*(x1-y1)/self.l)**2 + jnp.sin(jnp.pi*(x2-y2)/self.l)**2
        #return jnp.exp(-r/(2 * (self.sigma**2)))

    def D_x1_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 0)(x1, x2, y1, y2)
        return val

    def D_x2_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 1)(x1, x2, y1, y2)
        return val

    def DD_x1_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 0)(x1, x2, y1, y2)
        return val

    def DD_x2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 2)(x1, x2, y1, y2)
        return val

    def D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 3)(x1, x2, y1, y2)
        return val

    def DD_y1_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 2), 2)(x1, x2, y1, y2)
        return val

    def DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    def D_x1_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 2)(x1, x2, y1, y2)
        return val

    def D_x1_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 3)(x1, x2, y1, y2)
        return val

    def D_x1_DD_y1_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 0), 2), 2)(x1, x2, y1, y2)
        return val

    def D_x1_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 0), 3), 3)(x1, x2, y1, y2)
        return val

    def D_x2_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 1), 3)(x1, x2, y1, y2)
        return val

    def D_x2_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.D_x2_kappa, 2)(x1, x2, y1, y2)
        return val

    def D_x2_DD_y1_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 1), 2), 2)(x1, x2, y1, y2)
        return val

    def D_x2_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 1), 3), 3)(x1, x2, y1, y2)
        return val

    def DD_x1_DD_y1_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(grad(self.kappa, 0), 0), 2), 2)(x1, x2, y1, y2)
        return val

    def DD_x1_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(grad(self.kappa, 0), 0), 3), 3)(x1, x2, y1, y2)
        return val

    def DD_x2_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(grad(self.kappa, 1), 1), 3), 3)(x1, x2, y1, y2)
        return val

    def Delta_x_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 2), 2)(x1, x2, y1, y2)
        val += grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    def Delta_x_Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.Delta_y_kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.Delta_y_kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_x_Delta_x_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.Delta_x_kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.Delta_x_kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def D_x1_Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_y_kappa, 0)(x1, x2, y1, y2)
        return val

    def D_x2_Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_y_kappa, 1)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa, 2)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa, 3)(x1, x2, y1, y2)
        return val

    def radial_kappa(self, x1, x2):
        r1 = 1 / (self.l ** 2) * (jnp.cos(2 * jnp.pi * x1) - 1)
        r2 = 1 / (self.l ** 2) * (jnp.cos(2 * jnp.pi * x2) - 1)
        return jnp.exp(r1 + r2)

    def D_x1_radial_kappa(self, x1, x2):
        val = grad(self.radial_kappa, 0)(x1, x2)
        return val

    def D_x2_radial_kappa(self, x1, x2):
        val = grad(self.radial_kappa, 1)(x1, x2)
        return val

    def DD_x1_radial_kappa(self, x1, x2):
        val = grad(grad(self.radial_kappa, 0), 0)(x1, x2)
        return val

    def DD_x2_radial_kappa(self, x1, x2):
        val = grad(grad(self.radial_kappa, 1), 1)(x1, x2)
        return val



class Periodic_Matern_2D(object):
    def __init__(self, sigma, l, nu=2.5, nugget=1e-6):
        self.sigma = sigma  # Variance
        self.l = l  # Period
        self.nu = nu  # Smoothness parameter
        self.nugget = nugget  # Small nugget for numerical stability

    def _distance(self, x1, x2, y1, y2):
        r1 = 1 - jnp.cos(2 * jnp.pi * (x1 - y1) / self.l)
        r2 = 1 - jnp.cos(2 * jnp.pi * (x2 - y2) / self.l)
        return jnp.sqrt(r1 + r2 + self.nugget)

    def kappa(self, x1, x2, y1, y2):
        r = self._distance(x1, x2, y1, y2)
        scaling = jnp.sqrt(5) * r / self.sigma
        return (1 + scaling + (scaling ** 2) / 3) * jnp.exp(-scaling)

    def D_x1_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 0)(x1, x2, y1, y2)
        return val

    def D_x2_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 1)(x1, x2, y1, y2)
        return val

    def DD_x1_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 0)(x1, x2, y1, y2)
        return val

    def DD_x2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 2)(x1, x2, y1, y2)
        return val

    def D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(self.kappa, 3)(x1, x2, y1, y2)
        return val

    def DD_y1_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 2), 2)(x1, x2, y1, y2)
        return val

    def DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    def D_x1_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 2)(x1, x2, y1, y2)
        return val

    def D_x1_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 3)(x1, x2, y1, y2)
        return val

    def D_x1_DD_y1_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 0), 2), 2)(x1, x2, y1, y2)
        return val

    def D_x1_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 0), 3), 3)(x1, x2, y1, y2)
        return val

    def D_x2_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 1), 3)(x1, x2, y1, y2)
        return val

    def D_x2_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.D_x2_kappa, 2)(x1, x2, y1, y2)
        return val

    def D_x2_DD_y1_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 1), 2), 2)(x1, x2, y1, y2)
        return val

    def D_x2_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(self.kappa, 1), 3), 3)(x1, x2, y1, y2)
        return val

    def DD_x1_DD_y1_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(grad(self.kappa, 0), 0), 2), 2)(x1, x2, y1, y2)
        return val

    def DD_x1_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(grad(self.kappa, 0), 0), 3), 3)(x1, x2, y1, y2)
        return val

    def DD_x2_DD_y2_kappa(self, x1, x2, y1, y2):
        val = grad(grad(grad(grad(self.kappa, 1), 1), 3), 3)(x1, x2, y1, y2)
        return val

    def Delta_x_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.kappa, 2), 2)(x1, x2, y1, y2)
        val += grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2)
        return val

    def Delta_x_Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.Delta_y_kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.Delta_y_kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def Delta_x_Delta_x_kappa(self, x1, x2, y1, y2):
        val = grad(grad(self.Delta_x_kappa, 0), 0)(x1, x2, y1, y2)
        val += grad(grad(self.Delta_x_kappa, 1), 1)(x1, x2, y1, y2)
        return val

    def D_x1_Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_y_kappa, 0)(x1, x2, y1, y2)
        return val

    def D_x2_Delta_y_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_y_kappa, 1)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y1_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa, 2)(x1, x2, y1, y2)
        return val

    def Delta_x_D_y2_kappa(self, x1, x2, y1, y2):
        val = grad(self.Delta_x_kappa, 3)(x1, x2, y1, y2)
        return val

    def radial_kappa(self, x1, x2):
        r1 = 1 / (self.l ** 2) * (jnp.cos(2 * jnp.pi * x1) - 1)
        r2 = 1 / (self.l ** 2) * (jnp.cos(2 * jnp.pi * x2) - 1)
        return jnp.exp(r1 + r2)

    def D_x1_radial_kappa(self, x1, x2):
        val = grad(self.radial_kappa, 0)(x1, x2)
        return val

    def D_x2_radial_kappa(self, x1, x2):
        val = grad(self.radial_kappa, 1)(x1, x2)
        return val

    def DD_x1_radial_kappa(self, x1, x2):
        val = grad(grad(self.radial_kappa, 0), 0)(x1, x2)
        return val

    def DD_x2_radial_kappa(self, x1, x2):
        val = grad(grad(self.radial_kappa, 1), 1)(x1, x2)
        return val

