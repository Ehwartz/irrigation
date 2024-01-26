# -*- coding: utf-8 -*-
# @Author  : Ehwartz
# @Github  : https://github.com/Ehwartz
# @Time    : 12/21/2023
# @Software: PyCharm
# @File    : main.py

import torch
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm


# variables: d, r
def generate_grid(n, m, n_noz):
    xs = torch.linspace(-1.5, 1.5, n)
    ys = torch.linspace(-4, 4, m)
    grid_x, grid_y = torch.meshgrid([xs, ys], indexing='ij')
    grid_x = grid_x.reshape([n * m, 1])
    grid_y = grid_y.reshape([n * m, 1])
    # return torch.repeat_interleave(torch.concat([grid_x, grid_y], dim=1), repeats=n_noz, dim=0)
    return torch.concat([grid_x, grid_y], dim=1).repeat([n_noz, 1]).reshape([n_noz, n * m, 2]).permute([1, 0, 2])


def nozzle_grid(n_x=3, n_y=5) -> torch.Tensor:
    xs = torch.linspace(-3, 3, n_x)
    ys = torch.linspace(-8, 8, n_y)
    print('xs: ', xs)
    nozzle_x, nozzle_y = torch.meshgrid([xs, ys], indexing='ij')
    nozzle_x = nozzle_x.reshape([n_x * n_y, 1])
    nozzle_y = nozzle_y.reshape([n_x * n_y, 1])
    return torch.concat([nozzle_x, nozzle_y], dim=1)


def nozzle_coordinates(d: torch.Tensor, noz_grid: torch.Tensor) -> torch.Tensor:
    return d * noz_grid


def spray(r: torch.Tensor, noz_coordinates: torch.Tensor, grid):
    z = torch.square(r) - torch.sum(torch.square(noz_coordinates - grid), -1)
    mask = z > 0
    spray_grid_sum = (z * mask).sum(dim=-1)
    return spray_grid_sum


def forward(r, d, grid, noz_grid, alphas):
    noz_coordinates = nozzle_coordinates(d, noz_grid)
    n_grid = grid.size(0)
    z = torch.square(r) - torch.sum(torch.square(noz_coordinates - grid), -1)
    mask_p = z > 0
    covered = z[mask_p]
    spray_mean = covered.sum() / n_grid
    variance = (covered - spray_mean).square().mean()
    # mask_n = z < 0
    # uncovered = z[mask_n]
    # neg_spray = uncovered.abs().sqrt().sum()

    return -spray_mean.sqrt() * alphas[0] + variance.sqrt() * alphas[1] - d.abs() * 0.3  # + neg_spray * 0.001
    # return -spray_mean * alphas[0] + variance * alphas[1]


def plot_obj_func(rs, ds, grid, noz_grid, alphas):
    n_rs = rs.size(0)
    n_ds = ds.size(0)
    r_grid, d_grid = torch.meshgrid([rs, ds], indexing='ij')
    r_grid = r_grid.reshape([n_rs * n_ds])
    d_grid = d_grid.reshape([n_rs * n_ds])
    fs = []
    for i in tqdm(range(n_rs * n_ds)):
        # print(i)
        fs.append(forward(r_grid[i], d_grid[i], grid, noz_grid, alphas))

    f_grid = torch.tensor(fs).reshape([n_rs * n_ds, 1])
    figure = plt.figure()

    ax = figure.add_subplot(111, projection='3d')

    plt.xlabel("x")
    plt.ylabel("y")

    ax.plot_surface(r_grid.reshape([n_rs, n_ds]).detach().numpy(),
                    d_grid.reshape([n_rs, n_ds]).detach().numpy(),
                    f_grid.reshape([n_rs, n_ds]).detach().numpy(),
                    rstride=1, cstride=1, cmap="rainbow")
    plt.show()


def plot_spray(r, d, grid, noz_grid, grid_n, grid_m):
    print(noz_grid.size())
    print(grid.size())
    noz_coordinates = nozzle_coordinates(d, noz_grid)

    z = torch.square(r) - torch.sum(torch.square(noz_coordinates - grid), -1)
    mask = z > 0

    spray_grid_sum = (z * mask).sum(dim=-1)
    print(spray_grid_sum.size())
    grid0 = grid[:, 0, :]
    print(grid0.size())
    grid0x = grid0[:, 0].reshape([grid_n, grid_m]).detach().numpy()
    grid0y = grid0[:, 1].reshape([grid_n, grid_m]).detach().numpy()

    spray_grid = spray_grid_sum.reshape([grid_n, grid_m]).detach().numpy()

    figure = plt.figure()

    ax = figure.add_subplot(111, projection='3d')
    ax.set_box_aspect((3, 8, 0.8))
    plt.xlabel("x")
    plt.ylabel("y")

    ax.plot_surface(grid0x,
                    grid0y,
                    spray_grid,
                    rstride=1, cstride=1, cmap="rainbow")
    plt.show()


if __name__ == '__main__':
    r = torch.tensor([0.3], requires_grad=True)
    d = torch.tensor([0.3], requires_grad=True)
    n_x = 3
    n_y = 8
    noz_grid = nozzle_grid(n_x=n_x, n_y=n_y)
    grid_n = 31
    grid_m = 81
    grid = generate_grid(grid_n, grid_m, n_x * n_y)
    grid0 = grid[:, 0, :]

    # print(grid0.size())
    alphas = [1.0, 1.0]
    optimizer = torch.optim.Adam(params=[r, d], lr=1e-3)

    epoch = 3000
    for iep in tqdm(range(epoch)):
        def closure():
            with torch.enable_grad():
                optimizer.zero_grad()
                f = forward(r, d, grid, noz_grid, alphas)
                # print(f)
                f.backward()
            return f
        optimizer.step(closure=closure)
        print(f'r: {float(r)},  d: {float(d)}')

    print(d.detach()*noz_grid*10)

    rs = torch.linspace(0, 1.5, 51)
    ds = torch.linspace(0, 1, 51)
    with torch.no_grad():
        plot_obj_func(rs, ds, grid, noz_grid, alphas)

    plot_spray(r, d, grid, noz_grid, grid_n, grid_m)
