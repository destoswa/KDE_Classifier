import numpy as np
import torch
from scipy.stats import multivariate_normal
import random


class RandRotate(object):
    """ Rotate randomly
        Input:
        GxGxG array: the grid to be updated
    Output:
        GxGxG array: the rotated grid
    """
    def __call__(self, sample):
        num_rot = random.randint(0, 3)
        sample['grid'] = torch.rot90(sample['grid'], num_rot, (0, 1))

        return sample


class RandScale(object):
    """ Randomly scale patches of values in sample
        Input:
        GxGxG array: the grid to be updated
    Output:
        GxGxG array: the scaled grid
    """

    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def __call__(self, sample):
        idx_nonzeros = torch.nonzero(sample['grid'])
        num_candidates = torch.count_nonzero(sample['grid'])
        num_of_scales = random.randrange(0, int(num_candidates/self.kernel_size**3))
        scales = [random.uniform(0.5, 1.5) for i in range(num_of_scales)]

        # shuffle indices:
        idx = torch.randperm(idx_nonzeros.shape[0])
        idx_nonzeros = idx_nonzeros[idx, :]
        idx_nonzeros = idx_nonzeros[0:num_of_scales]

        # scales patches
        for id_point, point in enumerate(idx_nonzeros):
            sample['grid'][point[0] - self.kernel_size: point[0] + self.kernel_size,
            point[0] - self.kernel_size: point[0] + self.kernel_size,
            point[0] - self.kernel_size: point[0] + self.kernel_size] *= scales[id_point]

        return sample


class ToKDE(object):
    """ Convert pointCloud to KDE vector
        Input:
            Nx3 array where N is variable: the pointcloud
        Return:
            GxGxG array: the KDE grid
    """

    def __init__(self, grid_size, kernel_size, num_repeat):
        self.grid_size = grid_size
        self.kernel_size = kernel_size
        self.num_repeat = num_repeat

    def __call__(self, sample):
        pointCloud = sample['data']
        pointCloud = pcNormalize(pointCloud)

        # create grid:
        grid = pcToGrid(pointCloud, self.grid_size)

        # apply KDE:
        for rep in range(self.num_repeat):
            grid = applyKDE(grid, self.grid_size, self.kernel_size)

        # to tensor
        grid = torch.from_numpy(grid)
        label = torch.from_numpy(np.asarray(sample['label']))

        return {'data': grid,
                'label': label}


def pcToGrid(data, grid_size):
    """ Create a grid from the point cloud
        Input:
            Nx3 array where N is variable: the pointcloud
        Output:
            GxGxG array: the grid to be updated
    """
    grid = np.zeros((grid_size, grid_size, grid_size))

    # find position of each point on grid
    for id_point, point in enumerate(data):
        for idx, pos in enumerate(point):
            point[idx] = int((pos + 1)/2*grid_size)
        point = point.astype(int)

        # add point to grid
        grid[point[0], point[1], point[2]] = 1

    return grid


def applyKDE(grid, grid_size, kernel_size):
    """ Create a KDE grid from grid
        Input:
            GxGxG array: the grid to be updated
            3x1 array: the position of the point
            int : the size of the kernel
        Output:
            GxGxG array: the resulting KDE grid
    """
    # create kernel
    x, y, z = np.mgrid[-1:1.1:(1 / kernel_size), -1:1.1:(1 / kernel_size), -1:1.1:(1 / kernel_size)]
    pos = np.stack((x, y, z), axis=-1)
    rv = multivariate_normal([0, 0, 0])
    point_grid = rv.pdf(pos)*10

    # create the new grid to return (with margin to ease the application of the kernel on borders)
    new_grid = np.zeros((grid_size + 2 * kernel_size, grid_size + 2 * kernel_size, grid_size + 2 * kernel_size))

    # apply kernel on each non-null point of the grid
    lst_pos = np.where(grid != 0)
    for ind in range(len(lst_pos[0])):
        # find the value of the current point to scale the kernel
        coeff = grid[lst_pos[0][ind], lst_pos[1][ind], lst_pos[2][ind]]

        # add scaled values of the kernel centered at the position of the point
        new_grid[lst_pos[0][ind]: lst_pos[0][ind] + 2 * kernel_size + 1,
        lst_pos[1][ind]: lst_pos[1][ind] + 2 * kernel_size + 1,
        lst_pos[2][ind]: lst_pos[2][ind] + 2 * kernel_size + 1,
        ] += point_grid * coeff

    # return truncated new grid by removing the margin
    return new_grid[kernel_size:-kernel_size, kernel_size:-kernel_size, kernel_size:-kernel_size]


def pcNormalize(data):
    """ Normalize the data, use coordinates of the block centered at origin,
        Input:
            NxC array
        Output:
            NxC array
    """
    pc = data
    # print('derp')
    centroid = np.mean(data, axis=0)
    # print('durp')
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    # print(f"\t Max value: {m}")
    pc = pc / m if m > 0 else 0
    normal_data = pc
    return normal_data
