#import torch
#from functorch import vmap, grad
import jax.numpy as jnp
from jax import grad


class Anisotropic_Gaussian_kernel(object):
    def __init__(self):
        pass

    def kappa(self, x1, x2, y1, y2, params):
        scale_t = params[0]
        scale_x = params[1]
        r = ((x1 - y1) / scale_t) ** 2 + ((x2 - y2) / scale_x) ** 2
        return jnp.exp(-r)

    def D_x1_kappa(self, x1, x2, y1, y2, params):
        val = grad(self.kappa, 0)(x1, x2, y1, y2, params)
        return val

    def D_x2_kappa(self, x1, x2, y1, y2, params):
        val = grad(self.kappa, 1)(x1, x2, y1, y2, params)
        return val

    def DD_x2_kappa(self, x1, x2, y1, y2, params):
        val = grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2, params)
        return val

    def D_y1_kappa(self, x1, x2, y1, y2, params):
        val = grad(self.kappa, 2)(x1, x2, y1, y2, params)
        return val

    def D_y2_kappa(self, x1, x2, y1, y2, params):
        val = grad(self.kappa, 3)(x1, x2, y1, y2, params)
        return val

    def DD_y2_kappa(self, x1, x2, y1, y2, params):
        val = grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2, params)
        return val

    def D_x1_D_y1_kappa(self, x1, x2, y1, y2, params):
        val = grad(grad(self.kappa, 0), 2)(x1, x2, y1, y2, params)
        return val

    def D_x1_D_y2_kappa(self, x1, x2, y1, y2, params):
        val = grad(grad(self.kappa, 0), 3)(x1, x2, y1, y2, params)
        return val

    def D_x1_DD_y2_kappa(self, x1, x2, y1, y2, params):
        val = grad(grad(grad(self.kappa, 0), 3), 3)(x1, x2, y1, y2, params)
        return val

    def D_x2_D_y2_kappa(self, x1, x2, y1, y2, params):
        val = grad(grad(self.kappa, 1), 3)(x1, x2, y1, y2, params)
        return val

    def D_x2_D_y1_kappa(self, x1, x2, y1, y2, params):
        val = grad(self.D_x2_kappa, 2)(x1, x2, y1, y2, params)
        return val

    def D_x2_DD_y2_kappa(self, x1, x2, y1, y2, params):
        val = grad(grad(grad(self.kappa, 1), 3), 3)(x1, x2, y1, y2, params)
        return val

    def DD_x2_D_y1_kappa(self, x1, x2, y1, y2, params):
        val = grad(grad(grad(self.kappa, 1), 1), 2)(x1, x2, y1, y2, params)
        return val

    def DD_x2_D_y2_kappa(self, x1, x2, y1, y2, params):
        val = grad(grad(grad(self.kappa, 1), 1), 3)(x1, x2, y1, y2, params)
        return val

    def DD_x2_DD_y2_kappa(self, x1, x2, y1, y2, params):
        val = grad(grad(grad(grad(self.kappa, 1), 1), 3), 3)(x1, x2, y1, y2, params)
        return val

    def Delta_x_kappa(self, x1, x2, y1, y2, params):
        val = grad(grad(self.kappa, 0), 0)(x1, x2, y1, y2, params)
        val += grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2, params)
        return val

    def Delta_y_kappa(self, x1, x2, y1, y2, params):
        val = grad(grad(self.kappa, 2), 2)(x1, x2, y1, y2, params)
        val += grad(grad(self.kappa, 3), 3)(x1, x2, y1, y2, params)
        return val

    def Delta_x_y_kappa(self, x1, x2, y1, y2, params):
        val = grad(grad(self.Delta_y_kappa, 0), 0)(x1, x2, y1, y2, params)
        val += grad(grad(self.Delta_y_kappa, 1), 1)(x1, x2, y1, y2, params)
        return val

    def Delta_x_Delta_y_kappa(self, x1, x2, y1, y2, params):
        val = grad(grad(self.Delta_y_kappa, 0), 0)(x1, x2, y1, y2, params)
        val += grad(grad(self.Delta_y_kappa, 1), 1)(x1, x2, y1, y2, params)
        return val

    def Delta_x_D_y1_kappa(self, x1, x2, y1, y2, params):
        val = grad(self.Delta_x_kappa, 2)(x1, x2, y1, y2, params)
        return val

    def Delta_x_D_y2_kappa(self, x1, x2, y1, y2, params):
        val = grad(self.Delta_x_kappa, 3)(x1, x2, y1, y2, params)
        return val
    
    def D_x1_Delta_y_kappa(self, x1, x2, y1, y2, params):
        val = grad(self.Delta_y_kappa, 0)(x1, x2, y1, y2, params)
        return val
    
    def D_x2_Delta_y_kappa(self, x1, x2, y1, y2, params):
        val = grad(self.Delta_y_kappa, 1)(x1, x2, y1, y2, params)
        return val
