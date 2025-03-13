# ADMM implementation is based on the Barber + Sidky simulation http://arxiv.org/abs/2006.07278
# https://github.com/rinafb/ADMM_CT/blob/main/nonconvex_admm_CTsimulation.ipynb
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

from utils import *


def stop_criterion_iters(runningavg, idx, epsilon=1e-5):
    if idx < 101:
        return False
    # Stop when ||xbar_t - xbar_{t-1}|| < epsilon
    xbar_t = runningavg.get_xbar(idx)
    xbar_t1 = runningavg.get_xbar(idx-1)
    change = np.linalg.norm(xbar_t - xbar_t1)
    if idx == 1e3 or idx % 1e2 == 0:
        print(f'step {idx}: change is {change}, xbar_t has max value {max(xbar_t.flatten())}')
    return change < epsilon

def get_gradient_expterm(y,S):
    # returns l-by-m gradient of the first term in the negative log likelihood
    c_hat_by_energy_1 = get_c_hat_by_energy(y=y,S=S,deriv=1)
    return -(mu[:,None,None,:] * c_hat_by_energy_1).sum((1,3)).T

def get_gradient_logexpterm(y,c,S):
    # returns l-by-m gradient of the second term in the negative log likelihood
    c_hat_by_energy_1 = get_c_hat_by_energy(y=y,S=S,deriv=1)
    c_hat = get_c_hat(y=y,S=S)
    c_hat[c_hat==0] = 1 # avoiding 0/0
    return (mu[:,None,None,:] * (c/c_hat)[:,:,None]*c_hat_by_energy_1).sum((1,3)).T

def get_hessian_expterm(y, S):
    # returns l-by-m-by-m 2nd derivative of the first term in the negative log likelihood
    c_hat_by_energy_2 = get_c_hat_by_energy(y=y,S=S,deriv=2)
    return (c_hat_by_energy_2.sum(0)[:,None,:]*mu).dot(mu.T)



# ADMM update steps
def x_update(x, y, u, sig, P, nl, P_rowsums, P_colsums):
    return x + (1/sig)*(Ptmult(y=sig*(y-Pmult(x=x,P=P,nl=nl))/P_rowsums[:,None]-u, P=P))/P_colsums[:,:,None]

def y_update(x, y, u, c, sig, P, nl, S, P_rowsums):
    grad_y = lambda y_: get_gradient_expterm(y=y_,S=S) + get_gradient_logexpterm(y=y,c=c,S=S) \
                        + sig*(y_-Pmult(x=x,P=P,nl=nl))/P_rowsums[:,None] - u
    hess_y = lambda y_: get_hessian_expterm(y=y_,S=S) + sig/P_rowsums[:,None,None] * np.eye(nm)
    y_ = np.copy(y)
    for _ in range(niter_newton):
        y_ = y_ - np.linalg.solve(hess_y(y_),grad_y(y_))
    return y_

def u_update(x, y, u, sig, P, nl, P_rowsums):
    return u + sig*(Pmult(x=x, P=P, nl=nl)-y)/P_rowsums[:,None]

def run_admm(c, P, nl, S, maxtv=0):
    # maxtv denotes the oracle TV of the clean target image
    # maxtv=0 is a default to denote no TV projection (unregularized)
    P_rowsums, P_colsums = get_Psums(P, nl)
    x_store_iters = []
    runtimes = []
    for isig in tqdm(range(nsig)):
        runningavg = RunningAverage()
        sig = sig_grid[isig]
        x = np.zeros((nx,ny,nm))
        y = np.zeros((nl,nm))
        runningavg.update(np.copy(x))
        u = np.zeros((nl,nm))
        t0 = time.time()
        w_guess = 0.5
        it = 0
        while True:
            x = x_update(x=x, y=y, u=u, sig=sig, P=P, nl=nl, P_rowsums=P_rowsums, P_colsums=P_colsums)
            y = y_update(x=x, y=y, u=u, c=c, sig=sig, P=P, nl=nl, S=S, P_rowsums=P_rowsums)
            u = u_update(x=x, y=y, u=u, sig=sig, P=P, nl=nl, P_rowsums=P_rowsums)
            # do TV projection
            x, w_guess = project_tv_nonnegative(x, max_tv=maxtv, w_guess=w_guess)
            runningavg.update(np.copy(x))
            if (it + 1) % 100 == 0:
                # check stop criterion
                done = stop_criterion_iters(runningavg=runningavg, idx=it + 1)
                if done:
                    runtime = time.time() - t0
                    runtimes.append(runtime)
                    print(f'admm with sigma {sig} converged after {it + 1} steps, which took {runtime} seconds')
                    x_store_iters.append(runningavg)
                    break
            it = it + 1
    return np.array(x_store_iters), np.array(runtimes)



def run_mse_gd(c, P, nl, S, maxtv=0):
    # maxtv denotes the oracle TV of the clean target image
    # maxtv=0 is a default to denote no TV projection (unregularized)
    x_store_iters = []
    runtimes = []
    for istepsize in tqdm(range(ngdstepsize)):
        runningavg = RunningAverage()
        step_size = gdstepsizes[istepsize]
        x = np.zeros((nx,ny,nm))
        runningavg.update(np.copy(x))
        t0 = time.time()
        it = 0
        w_guess = 0.5
        while True:
            direction = get_L2_direction(x=x, c=c, S=S, P=P, nl=nl)
            x = x - step_size * direction
            # do TV projection
            x, w_guess = project_tv_nonnegative(x, max_tv=maxtv, w_guess=w_guess)
            runningavg.update(np.copy(x))
            if (it + 1) % 100 == 0:
                # check stop criterion
                done = stop_criterion_iters(runningavg=runningavg, idx=it + 1)
                if done:
                    runtime = time.time() - t0
                    runtimes.append(runtime)
                    print(f'MSE GD with stepsize {step_size} converged after {it + 1} steps, which took {runtime} seconds')
                    x_store_iters.append(runningavg)
                    break
            it = it + 1
    return np.array(x_store_iters), np.array(runtimes)



# Polyak SGM (L1 error minimization)
def get_L1direction(x,c,S,P,nl):
    c_hat = get_c_hat(y=Pmult(x=x,P=P,nl=nl), S=S)
    errorsign = np.sign(c - c_hat)  # [nw, nl]
    a = P.reshape(nl, -1)  # [nl, nx*ny]
    expterm = qexp(-np.dot(Pmult(x,P,nl),mu))  # [nl, ni]
    direction = 0
    for w in range(nw):
        for l in range(nl):
            direction = direction + errorsign[w,l] * a[l].reshape(nx*ny,1) @ (expterm[l].reshape(1, ni) @ np.diag(S[w,l,:]) @ mu.T)  # [nx*ny, nm]
    return direction.reshape(nx, ny, nm) / nl

def polyakSGMstep(x, c_star, step_size, P, nl, S, fstar):
    v = get_L1direction(x=x, c=c_star, S=S, P=P, nl=nl)
    scaling = (get_L1loss(x=x, c=c_star, S=S, P=P, nl=nl) - fstar) / np.sum(np.square(v))
    return x - step_size * scaling * v

# PolyakSGM: Algorithm 1 in https://arxiv.org/pdf/2407.12984
# Since there is measurement noise we cannot assume the optimal loss is zero. 
# Here we use the oracle value of the loss incurred by the ground truth image, although this might not be the truly minimal loss possible.
def run_polyaksgm(c, P, nl, S, xstar, maxtv=0):
    # maxtv denotes the oracle TV of the clean target image
    # maxtv=0 is a default to denote no TV projection (unregularized)
    fstar_oracle = get_L1loss(x=xstar, c=c, S=S, P=P, nl=nl)
    x_store_iters = []
    runtimes = []
    w_guess = 0.5
    for istepsize in tqdm(range(npolyakstepsize)):
        runningavg = RunningAverage()
        step_size = polyakstepsizes[istepsize]
        x = np.zeros((nx,ny,nm))
        runningavg.update(np.copy(x))
        t0 = time.time()
        it = 0
        while True:
            x = polyakSGMstep(x=x, c_star=c, step_size=step_size, P=P, nl=nl, S=S, fstar=fstar_oracle)
            # do TV projection
            x, w_guess = project_tv_nonnegative(x, max_tv=maxtv, w_guess=w_guess)
            runningavg.update(np.copy(x))
            if (it + 1) % 100 == 0:
                # check stop criterion
                done = stop_criterion_iters(runningavg=runningavg, idx=it + 1)
                if done:
                    runtime = time.time() - t0
                    runtimes.append(runtime)
                    print(f'polyakSGM_oracle with stepsize {step_size} converged after {it + 1} steps, which took {runtime} seconds')
                    x_store_iters.append(runningavg)
                    break
            it = it + 1
    # x_store_iters = np.array(x_store_iters)  # array of nstepsize arrays where each one is shape [niter, nx, ny, nm]
    return np.array(x_store_iters), np.array(runtimes)



# Monotone operator
def take_monotone_step(x, c_star, step_size, P, nl, S):
    direction = get_monotone_direction(x=x, c_star=c_star, P=P, nl=nl, S=S)
    return x - step_size * direction

def run_monotone(c, P, nl, S, maxtv=0, total_intensity=1e6):
    # maxtv denotes the oracle TV of the clean target image
    # maxtv=0 is a default to denote no TV projection (unregularized)
    x_store_iters = []
    runtimes = []
    for istepsize in tqdm(range(nstepsize)):
        runningavg = RunningAverage()
        step_size = stepsizes[istepsize]
        # Scale the stepsize by total intensity if needed
        scale_factor = 1e6 / total_intensity
        step_size = step_size * scale_factor
        print(f'monotone is actually using stepsize {step_size} for {total_intensity} photons')
        x = np.zeros((nx,ny,nm))
        runningavg.update(np.copy(x))
        t0 = time.time()
        it = 0
        w_guess = 0.5
        while True:
            x = take_monotone_step(x=x, c_star=c, step_size=step_size, P=P, nl=nl, S=S)
            # do TV projection
            x, w_guess = project_tv_nonnegative(x, max_tv=maxtv, w_guess=w_guess)
            runningavg.update(np.copy(x))
            if (it + 1) % 100 == 0:
                # check stop criterion
                done = stop_criterion_iters(runningavg=runningavg, idx=it + 1)
                if done:
                    runtime = time.time() - t0
                    runtimes.append(runtime)
                    print(f'monotone with stepsize {step_size} converged after {it + 1} steps, which took {runtime} seconds')
                    x_store_iters.append(runningavg)
                    break
            it = it + 1
    return np.array(x_store_iters), np.array(runtimes)



# y = proj(x - \eta F(x))
# x_+ = proj(x - \eta F(y))
def take_extragradient_step(x, c_star, step_size, P, nl, S, max_tv, w_guess=0.5):
    direction = get_monotone_direction(x=x, c_star=c_star, P=P, nl=nl, S=S)
    y, w_guess = project_tv_nonnegative(x - step_size * direction, max_tv=max_tv, w_guess=w_guess)
    y = x - step_size * direction
    direction = get_monotone_direction(x=y, c_star=c_star, P=P, nl=nl, S=S)
    return project_tv_nonnegative(x - step_size * direction, max_tv=max_tv, w_guess=w_guess) 

def run_extragradient(c, P, nl, S, maxtv=0, total_intensity=1e6):
    x_store_iters = []
    runtimes = []
    for istepsize in tqdm(range(nstepsize)):
        runningavg = RunningAverage()
        step_size = stepsizes[istepsize]
        # Scale the stepsize by total intensity if needed
        scale_factor = 1e6 / total_intensity
        step_size = step_size * scale_factor
        print(f'extragradient is actually using stepsize {step_size} for {total_intensity} photons')
        x = np.zeros((nx,ny,nm))
        runningavg.update(np.copy(x))
        t0 = time.time()
        it = 0
        w_guess = 0.5
        while True:
            x, w_guess = take_extragradient_step(x=x, c_star=c, step_size=step_size, P=P, nl=nl, S=S, max_tv=maxtv, w_guess=w_guess)
            runningavg.update(np.copy(x))
            if (it + 1) % 100 == 0:
                done = stop_criterion_iters(runningavg=runningavg, idx=it + 1)
                if done:
                    runtime = time.time() - t0
                    runtimes.append(runtime)
                    print(f'extragradient with stepsize {step_size} converged after {it + 1} steps, which took {runtime} seconds')
                    x_store_iters.append(runningavg)
                    break
            it = it + 1
    return np.array(x_store_iters), np.array(runtimes)





