import numpy as np
import typing
#np.set_printoptions(precision=20)

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

class Interval(object):
    def __init__(self, xl, xr):
        self.xl = xl
        self.xr = xr

    def random_seed(self, s):
        np.random.seed(s)

    def sampling(self, M, M_Omega):
        int_points = np.random.uniform(self.xl, self.xr, M_Omega)
        bdry_points = np.array([self.xl, self.xr])
        points = np.concatenate((int_points, bdry_points))
        return points

class Square(object):
    def __init__(self, x1l, x1r, x2l, x2r):
        self.x1l = x1l
        self.x1r = x1r
        self.x2l = x2l
        self.x2r = x2r

    def random_seed(self, s):
        np.random.seed(s)

    def sampling(self, M, M_Omega):
        int_points = np.concatenate((np.random.uniform(self.x1l, self.x1r, (M_Omega, 1)), np.random.uniform(self.x2l, self.x2r, (M_Omega, 1))), axis=1)
        bdry_points = np.zeros((M - M_Omega, 2))
        num_per_bdry = int((M - M_Omega) / 4)

        # bottom face
        bdry_points[0:num_per_bdry, 0] = np.random.uniform(self.x1l, self.x1r, num_per_bdry)
        bdry_points[0:num_per_bdry, 1] = self.x2l
        # right face
        bdry_points[num_per_bdry:2 * num_per_bdry, 0] = self.x1r
        bdry_points[num_per_bdry:2 * num_per_bdry, 1] = np.random.uniform(self.x2l, self.x2r, num_per_bdry)
        # top face
        bdry_points[2 * num_per_bdry:3 * num_per_bdry, 0] = np.random.uniform(self.x1l, self.x1r, num_per_bdry)
        bdry_points[2 * num_per_bdry:3 * num_per_bdry, 1] = self.x2r
        # left face
        bdry_points[3 * num_per_bdry:M - M_Omega, 1] = np.random.uniform(self.x2l, self.x2r, M - M_Omega - 3 * num_per_bdry)
        bdry_points[3 * num_per_bdry:M - M_Omega, 0] = self.x1l

        points = np.concatenate((int_points, bdry_points), axis=0)
        return points

class TimeDependentR2DSquare(object):
    def __init__(self, ti, te, x1l, x1r, x2l, x2r):
        self.ti = ti
        self.te = te
        self.x1l = x1l
        self.x1r = x1r
        self.x2l = x2l
        self.x2r = x2r

    def random_seed(self, s):
        np.random.seed(s)

    def sampling(self, M_Omega, M_I, M_T):
        int_points = np.concatenate((np.random.uniform(self.ti, self.te, (M_Omega, 1)), np.random.uniform(self.x1l, self.x1r, (M_Omega, 1)), np.random.uniform(self.x2l, self.x2r, (M_Omega, 1))), axis=1)
        initial_points = np.concatenate((self.ti + np.zeros((M_I, 1)), np.random.uniform(self.x1l, self.x1r, (M_I, 1)), np.random.uniform(self.x2l, self.x2r, (M_I, 1))), axis=1)
        terminal_points = np.concatenate((self.te + np.zeros((M_T, 1)), np.random.uniform(self.x1l, self.x1r, (M_T, 1)), np.random.uniform(self.x2l, self.x2r, (M_T, 1))), axis=1)
        points = np.concatenate((int_points, initial_points, terminal_points), axis=0)
        return points

    def sampling_at_time_t(self, t, M):
        points = np.concatenate((t + np.zeros((M, 1)), np.random.uniform(self.x1l, self.x1r, (M, 1)), np.random.uniform(self.x2l, self.x2r, (M, 1))), axis=1)
        return points

class TimeDependentR1DSquare(object):
    def __init__(self, ti, te, x1l, x1r):
        self.ti = ti
        self.te = te
        self.x1l = x1l
        self.x1r = x1r

    def random_seed(self, s):
        np.random.seed(s)

    @typing.overload
    def sampling(self, M_Omega, M_I, M_T):
        int_points = np.concatenate((np.random.uniform(self.ti, self.te, (M_Omega, 1)), np.random.uniform(self.x1l, self.x1r, (M_Omega, 1))), axis=1)
        initial_points = np.concatenate((self.ti + np.zeros((M_I, 1)), np.random.uniform(self.x1l, self.x1r, (M_I, 1))), axis=1)
        terminal_points = np.concatenate((self.te + np.zeros((M_T, 1)), np.random.uniform(self.x1l, self.x1r, (M_T, 1))), axis=1)
        points = np.concatenate((int_points, initial_points, terminal_points), axis=0)
        return points

    def sampling(self, M_Omega, M_I):
        int_points = np.concatenate((np.random.uniform(self.ti, self.te, (M_Omega, 1)), np.random.uniform(self.x1l, self.x1r, (M_Omega, 1))), axis=1)
        initial_points = np.concatenate((self.ti + np.zeros((M_I, 1)), np.random.uniform(self.x1l, self.x1r, (M_I, 1))), axis=1)
        points = np.concatenate((int_points, initial_points), axis=0)
        return points

    # def sampling(self, M_Omega, M_I, M_T):
    #     N = 30
    #     h = 4 / N
    #     XX = np.linspace(-2 + h / 2, 2 - h / 2, N)
    #     SEG = 30
    #     dms = XX
    #     dt = 1 / SEG
    #     ts = np.linspace(dt, self.te, SEG)  # jnp.concatenate((jnp.arange(0, SEG) / SEG, jnp.array([1])))
    #     XX2d, YY2d = np.meshgrid(ts, dms)
    #     int_points = np.concatenate((XX2d.reshape(-1, 1), YY2d.reshape(-1, 1)), axis=1)
    #
    #     # int_points = np.concatenate((np.random.uniform(self.ti, self.te, (M_Omega, 1)), np.random.uniform(self.x1l, self.x1r, (M_Omega, 1))), axis=1)
    #     initial_points = np.concatenate((self.ti + np.zeros((M_I, 1)), np.random.uniform(self.x1l, self.x1r, (M_I, 1))), axis=1)
    #     terminal_points = np.concatenate((self.te + np.zeros((M_T, 1)), np.random.uniform(self.x1l, self.x1r, (M_T, 1))), axis=1)
    #     points = np.concatenate((int_points, initial_points, terminal_points), axis=0)
    #
    #     l, r = int_points.shape
    #     return points, l

    def sampling_at_time_t(self, t, M):
        points = np.concatenate((t + np.zeros((M, 1)), np.random.uniform(self.x1l, self.x1r, (M, 1))), axis=1)
        return points


class TimeDependentR1D(object):
    def __init__(self, ti, te, x1l, x1r):
        self.ti = ti
        self.te = te
        self.x1l = x1l
        self.x1r = x1r

    def random_seed(self, s):
        np.random.seed(s)

    def sampling(self, M_time, M_Omega):
        time_points = np.linspace(self.ti, self.te, M_time)
        # points at the initial time
        points_i = np.concatenate((self.ti + np.zeros((M_Omega, 1)), np.random.uniform(self.x1l, self.x1r, (M_Omega, 1))), axis=1)
        # points at the terminal time
        points_e = np.concatenate((self.te + np.zeros((M_Omega, 1)), np.random.uniform(self.x1l, self.x1r, (M_Omega, 1))), axis=1)
        points = np.concatenate((points_i, points_e), axis=0)
        for t in reversed(time_points[1:M_time-1]):
            points_t = np.concatenate((t + np.zeros((M_Omega, 1)), np.random.uniform(self.x1l, self.x1r, (M_Omega, 1))), axis=1)
            points = np.concatenate((points_t, points), axis=0)

        return points

    def sampling_at_time_t(self, t, M):
        points = np.concatenate((t + np.zeros((M, 1)), np.random.uniform(self.x1l, self.x1r, (M, 1))), axis=1)
        return points

class TimeDependentR2D(object):
    def __init__(self, ti, te, x1l, x1r, x2l, x2r):
        self.ti = ti
        self.te = te
        self.x1l = x1l
        self.x1r = x1r
        self.x2l = x2l
        self.x2r = x2r

    def random_seed(self, s):
        np.random.seed(s)

    def sampling(self, M_time, M_Omega):
        time_points = np.linspace(self.ti, self.te, M_time)
        # points at the initial time
        points_i = np.concatenate((self.ti + np.zeros((M_Omega, 1)),
                                   np.random.uniform(self.x1l, self.x1r, (M_Omega, 1)),
                                   np.random.uniform(self.x2l, self.x2r, (M_Omega, 1))), axis=1)
        # points at the terminal time
        points_e = np.concatenate((self.te + np.zeros((M_Omega, 1)),
                                   np.random.uniform(self.x1l, self.x1r, (M_Omega, 1)),
                                   np.random.uniform(self.x2l, self.x2r, (M_Omega, 1))), axis=1)
        points = np.concatenate((points_i, points_e), axis=0)
        for t in reversed(time_points[1:M_time-1]):
            points_t = np.concatenate((t + np.zeros((M_Omega, 1)),
                                         np.random.uniform(self.x1l, self.x1r, (M_Omega, 1)),
                                         np.random.uniform(self.x2l, self.x2r, (M_Omega, 1))), axis=1)
            points = np.concatenate((points_t, points), axis=0)

        return points
        # int_points = np.concatenate((np.random.uniform(self.ti, self.te, (M_Omega, 1)), np.random.uniform(self.x1l, self.x1r, (M_Omega, 1)), np.random.uniform(self.x2l, self.x2r, (M_Omega, 1))), axis=1)
        # initial_points = np.concatenate((self.ti + np.zeros((M_I, 1)), np.random.uniform(self.x1l, self.x1r, (M_I, 1)), np.random.uniform(self.x2l, self.x2r, (M_I, 1))), axis=1)
        # terminal_points = np.concatenate((self.te + np.zeros((M_T, 1)), np.random.uniform(self.x1l, self.x1r, (M_T, 1)), np.random.uniform(self.x2l, self.x2r, (M_T, 1))), axis=1)
        # points = np.concatenate((int_points, initial_points, terminal_points), axis=0)
        #return points

    def sampling_at_time_t(self, t, M):
        points = np.concatenate((t + np.zeros((M, 1)), np.random.uniform(self.x1l, self.x1r, (M, 1)), np.random.uniform(self.x2l, self.x2r, (M, 1))), axis=1)
        return points

class TimeDependentSquare(object):
    def __init__(self, x1l, x1r, x2l, x2r):
        self.x1l = x1l
        self.x1r = x1r
        self.x2l = x2l
        self.x2r = x2r

    def random_seed(self, s):
        np.random.seed(s)

    def sampling(self, M, M_Omega):
        int_points = np.concatenate((np.random.uniform(self.x1l, self.x1r, (M_Omega, 1)), np.random.uniform(self.x2l, self.x2r, (M_Omega, 1))), axis=1)
        bdry_points = np.zeros((M - M_Omega, 2))
        num_per_bdry = int((M - M_Omega) / 3)
        #bdry_points = np.zeros((num_per_bdry * 3, 2))
        #num_per_bdry = M - M_Omega

        # bottom face
        bdry_points[0:num_per_bdry, 0] = self.x1l
        bdry_points[0:num_per_bdry, 1] = np.random.uniform(self.x2l, self.x2r, num_per_bdry)
        # # right face
        bdry_points[num_per_bdry:2 * num_per_bdry, 0] = np.random.uniform(self.x1l, self.x1r, num_per_bdry)
        bdry_points[num_per_bdry:2 * num_per_bdry, 1] = self.x2r
        # left face
        bdry_points[2 * num_per_bdry:, 0] = np.random.uniform(self.x1l, self.x1r, M - M_Omega - 2 * num_per_bdry)
        bdry_points[2 * num_per_bdry:, 1] = self.x2l

        points = np.concatenate((int_points, bdry_points), axis=0)
        return points


class Torus1D(object):
    def __init__(self):
        pass

    def random_seed(self, s):
        np.random.seed(s)

    def sampling(self, M, M_Omega):
        points = np.random.uniform(0, 1, M_Omega)
        # points = np.append(points, 0)
        # points = np.append(points, 1)
        return points

class Torus2D(object):
    def __init__(self):
        pass

    def random_seed(self, s):
        np.random.seed(s)

    def sampling(self, M_Omega):
        points = np.concatenate((np.random.uniform(0, 1, (M_Omega, 1)), np.random.uniform(0, 1, (M_Omega, 1))), axis=1)
        return points



class TimeDependentTorus1D(object):
    def __init__(self, ti, te, x1l, x1r):
        self.ti = ti
        self.te = te
        self.x1l = x1l
        self.x1r = x1r

    def random_seed(self, s):
        np.random.seed(s)

    def sampling(self, M, M_Omega):
        int_points = np.concatenate((np.random.uniform(self.ti, self.te, (M_Omega, 1)), np.random.uniform(self.x1l, self.x1r, (M_Omega, 1))), axis=1)
        initial_points = np.concatenate((self.ti + np.zeros((M - M_Omega, 1)), np.random.uniform(self.x1l, self.x1r, (M - M_Omega, 1))), axis=1)
        points = np.concatenate((int_points, initial_points), axis=0)
        return points

    def sampling_at_time_t(self, t, M):
        points = np.concatenate((t + np.zeros((M, 1)), np.random.uniform(self.x1l, self.x1r, (M, 1))), axis=1)
        return points


