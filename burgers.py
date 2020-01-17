
# coding: utf-8

# In[1]:


# Solving burgers' equation with forward finite difference in time,
# and central difference in space.
# u_t + u * u_x = 0
# u_t + ((u**2)/2)_x = 0
# u_t = -1/2 * d(u**2)/dx


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


# In[3]:


def simulate(end_time, timesteps, xs, initial, left, right):
    space_steps = len(initial)
    dt = end_time / timesteps
    results = np.zeros((timesteps, space_steps), dtype=np.float32)
    results[0] = initial
    # TODO: kinda jank but I guess its actually ok
    dxs = np.array([xs[i] - xs[i - 1] for i in range(1, space_steps)])
    dxs = np.array([xs[i] - xs[i - 2] for i in range(2, space_steps)])
    k = dt / 2
    for i in range(1, timesteps):
        left_u = results[i-1, :-2]
        right_u = results[i-1, 2:]
        results[i, 1:space_steps - 1] = (left_u + right_u) / 2 + k * (left_u**2 - right_u**2) / dxs
#         for j in range(1, space_steps - 1):
#             left_u = results[i-1, j-1]
#             right_u = results[i-1, j+1]
#             results[i, j] = (left_u + right_u) / 2 + dt * (left_u**2 - right_u**2) / (4 * dxs[j - 1])
        results[i, 0] = left(dt * i)
        results[i, -1] = right(dt * i)
    return results

import random

def random_poly(order, coef_stdev):
    if (order == 0):
        return (lambda x: 1)
    if (order == 1):
        return (lambda x: 1 - x)
    prev = random_poly(order - 1, coef_stdev)
    k = random.gauss(0, coef_stdev / np.sqrt(order))
    return (lambda x: (1 + k*x) * prev(x))

def random_poly_flip(order, coef_stdev):
    if (order == 0):
        return (lambda x: 1)
    if (order == 1):
        return (lambda x: x)
    prev = random_poly_flip(order - 1, coef_stdev)
    k = random.gauss(0, coef_stdev / np.sqrt(order))
    return (lambda x: (1 + k * (1 - x)) * prev(x))

def piecewise_random_split(xs, max_splits, max_degree, coef_stdev, jump_chance):
    num_splits = random.randint(0, max_splits)
    return piecewise_random_poly(xs, num_splits, max_degree, coef_stdev, jump_chance)
    
def piecewise_random_poly(xs, num_splits, max_degree, coef_stdev, jump_chance):
    split_locs = np.random.choice(len(xs) - 2, size=num_splits, replace=False)
    split_locs.sort()
    split_locs = list(split_locs)
    split_locs.append(len(xs) - 1)
    
    i0 = 0
    start_y = random.gauss(0, coef_stdev)
    
    ys = np.zeros(len(xs))
    
    for i1 in split_locs:
        poly_degree = random.randint(0, max_degree)
        inputs = np.linspace(0, 1, num=i1-i0+1)
        if random.random() < 0.5:
            poly = random_poly_flip(poly_degree, coef_stdev)
            ys[i0:i1 + 1] = start_y + random.gauss(0, coef_stdev) * poly(inputs)
        else:
            poly = random_poly(poly_degree, coef_stdev)
            scaling = random.gauss(0, coef_stdev)
            ys[i0:i1 + 1] = start_y + scaling * (poly(inputs) - 1)
        
        i0 = i1
        if random.random() < jump_chance:
            start_y = random.gauss(0, coef_stdev)
        elif i1 < len(xs):
            start_y = ys[i1]
    return ys

def const_x(x):
    return (lambda _: x)


# In[6]:


def run_many_and_save(num_runs, dest_file, save_full=False):
    xs = np.linspace(0, 1, num=400)
    t_steps = 800
    end_t = 3
    results_list = []
    results_labels = []
    successes = 0
    for i in range(num_runs):
        if i % 100 == 0:
            print("Running simulation number ", i)
        ys = piecewise_random_poly(xs, 2, 5, 0.5, 0.25)
        results = simulate(end_t, t_steps, xs, ys, lambda x: ys[0], lambda x: ys[-1])
        success = np.isfinite(results).all()
        successes += success
        if (save_full):
            results_list.append((results[0], results, success))
        else:
            results_list.append(results[0])
            results_labels.append(success)
    results_array = np.array(results_list)
    np.save(dest_file, results_array)    
    if not save_full:
        labels_array = np.array(results_labels)
        np.save(dest_file + "_labels", labels_array)
    print("done!")
    print("Successes: {}, runs: {}".format(successes, num_runs))


# In[7]:


# all_results = run_many_and_save(1000, "out.npy", save_full=True)


# In[ ]:


# run_many_and_save(10000, "burgers_50k-1.npy")
# run_many_and_save(10000, "burgers_50k-2.npy")
# run_many_and_save(10000, "burgers_50k-3.npy")
# run_many_and_save(10000, "burgers_50k-4.npy")
# run_many_and_save(10000, "burgers_50k-5.npy")

run_many_and_save(10000, "burgers_testing_only.npy")

# In[ ]:


# print(training_results[:10])


# In[ ]:


# xs = np.linspace(0, 1, num=400)
# for i in range(10):
    # plt.figure()
    # plt.plot(xs, training_results[i][0])
    # print(training_results[i][1])
    # plt.show()

