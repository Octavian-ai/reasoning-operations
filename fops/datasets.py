

import tensorflow as tf
import numpy as np
from collections import namedtuple

n_train = 1_000_000
n_test = 4_000
bus_width = 128

Dataset = namedtuple('Dataset', ['train', 'test'])

# Expected shape of dataset data: [n_items, 2, bus_width]


class hashable_ndarray(object):
	def __init__(self, v):
		self.v = v

	def __hash__(self):
		self.v.flags.writeable = False
		return hash(self.v.data.tobytes())


def int_to_one_hot(arr, n_classes):
    """

    :param arr: N dim array of size i_1, ..., i_N
    :param n_classes: C
    :returns: one-hot N+1 dim array of size i_1, ..., i_N, C
    :rtype: ndarray

    """
    arr = np.array(arr)
    assert arr.size > 0, "Cannot hotify empty array"
    one_hot = np.zeros(arr.shape + (n_classes,))
    axes_ranges = [range(arr.shape[i]) for i in range(arr.ndim)]
    flat_grids = [_.ravel() for _ in np.meshgrid(*axes_ranges, indexing='ij')]
    one_hot[tuple(flat_grids + [arr.ravel()])] = 1
    assert((one_hot.sum(-1) == 1).all())
    assert(np.allclose(np.argmax(one_hot, -1), arr))
    return one_hot


def gen_one_hot():

	unique_pairs = set()

	while len(unique_pairs) < n_train + n_test:
		unique_pairs.add(hashable_ndarray(np.random.randint(0, bus_width, 2)))

	unique_pairs = list(unique_pairs)
	unique_pairs = [i.v for i in unique_pairs]

	train = int_to_one_hot(unique_pairs[0:n_train], bus_width)
	test  = int_to_one_hot(unique_pairs[n_train:][:n_test], bus_width)

	return Dataset(train, test)



def gen_many_hot():

	unique_pairs = set()

	while len(unique_pairs) < n_train + n_test:
		unique_pairs.add(hashable_ndarray(np.random.randint(0, 2, (2, bus_width))))

	unique_pairs = list(unique_pairs)
	unique_pairs = [i.v for i in unique_pairs]

	train = np.array(unique_pairs[0:n_train]).astype(np.float32)
	test  = np.array(unique_pairs[n_train:][:n_test]).astype(np.float32)

	return Dataset(train, test)



def gen_random():
	train = np.random.random_sample((n_train, 2, bus_width)) * 2.0 - 1.0
	test = np.random.random_sample((n_test, 2, bus_width)) * 2.0 - 1.0
	return Dataset(train, test)

def gen_random_pos():
	train = np.random.random_sample((n_train, 2, bus_width))
	test = np.random.random_sample((n_test, 2, bus_width))
	return Dataset(train, test)



datasets = {
	"one_hot": gen_one_hot,
	"many_hot": gen_many_hot,
	"random_pos": gen_random,
	"random": gen_random,
}

