import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random, grad
import math
import flax
from flax import linen as nn
from typing import Any, Callable


class LengthScaleNetwork2D(nn.Module):
    """Fully-connected neural network."""

    layer_sizes: Any
    params: Any = None

    def setup(self):
        self.denses = [
            nn.Dense(
                unit,
                param_dtype=jnp.float64,
            )
            for unit in self.layer_sizes[1:]
        ]
        # self._activation = jax.nn.relu
        self._activation = jnp.tanh

    def __call__(self, inputs, training=False):
        x = inputs
        # print("Inputs shape: ")
        # print(x.shape)
        for j, linear in enumerate(self.denses[:-1]):
            x = self._activation(linear(x))
        x = self.denses[-1](x)
        x = jax.nn.softplus(x)
        # print("Output shape:")
        # print(x.shape)
        return x / 4
    
    
class GibbsKernel2D:

    def __init__(self, ls_net):
        self.ls_net = ls_net
        self.output_fn = jax.jit(
            lambda params, inputs: self.ls_net.apply(params, inputs)
        )

    def kappa(self, x1, x2, y1, y2, params):
        input_x = jnp.stack([x1, x2])
        input_y = jnp.stack([y1, y2])
        # out_x = self.ls_net.apply(params, input_x)
        # out_y = self.ls_net.apply(params, input_y)
        out_x = self.output_fn(params, input_x)
        out_y = self.output_fn(params, input_y)


        out_x1, out_x2 = jnp.split(out_x, 2)
        out_y1, out_y2 = jnp.split(out_y, 2)

        # for additional cases
        out_x2 = out_x2 / 1
        out_y2 = out_y2 / 1

        k1 = jnp.exp(-(x1 - y1)**2 / (out_x1**2 + out_y1**2))
        k2 = jnp.exp(-(x2 - y2)**2 / (out_x2**2 + out_y2**2))
        k1 = k1 * jnp.sqrt(2*out_x1*out_y1 / (out_x1**2 + out_y1**2))
        k2 = k2 * jnp.sqrt(2*out_x2*out_y2 / (out_x2**2 + out_y2**2))
        k = (k1 * k2).reshape([])
        return k

    # def kappa(self, x1, x2, y1, y2, params):
    #     # for the Burgers' equation
    #     lx = 1 / jnp.sqrt(2) / 20.0
    #     lt = 1 / jnp.sqrt(2) / 3.0

    #     dx2 = (x2 - y2)**2
    #     dt2 = (x1 - y1)**2
    #     kx = jnp.exp(-dx2 / 2 / lx**2)
    #     kt = jnp.exp(-dt2 / 2 / lt**2)
    #     return kx * kt

    # def kappa(self, x1, x2, y1, y2, params):
    #     # for the Eikonal equation
    #     lx = 0.2
    #     ly = 0.2

    #     dx2 = (x1 - y1)**2
    #     dy2 = (x2 - y2)**2
    #     kx = jnp.exp(-dx2 / 2 / lx**2)
    #     ky = jnp.exp(-dy2 / 2 / ly**2)
    #     return kx * ky

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


class KernelGenerator:

    def __init__(self, ls_net: LengthScaleNetwork2D):
        self.ls_net = ls_net

    def create_initial_params(self, key):
        _inputs = jnp.ones(shape=[3, 2])
        params = self.ls_net.init(key, _inputs)
        return params

    def __call__(self):
        return GibbsKernel2D(self.ls_net)
