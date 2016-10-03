#!/usr/bin/python

import numpy as np
import argparse
from datetime import datetime

from matplotlib import pyplot as plt
from scipy.spatial.distance import cityblock
from scipy.spatial.kdtree import KDTree

def random_voronoi(size, num_points, use_brute=False, verbose=False, out_file="out.png"):
    """
    Method that generates random points in a size x size square, then displays the Voronoi regions
    for these random points in the square.
    :param size: the size length of the square.
    :param num_points: the number of random points to generate.
    :param use_brute: a boolean specifying whether a Scipy KDTree should be used to speed calculations.
    :param verbose: an integer specifying the level of debugging information to be output.
    :param out_file: a string specifying an output file to save generated images to.
    :return: None
    """
    sources = []

    # Generate random points
    for i in range(num_points):
        temp = [np.random.randint(0, size) for j in range(2)]
        sources.append((temp[0], temp[1]))

    im = np.zeros((size, size))
    im2 = np.zeros((size, size))

    for source in sources:
        im[source] = 255
        im2[source] = 255

    start_time = datetime.now()

    # Use KDTree to find the closest source point to each pixel, and set pixel to appropriate color
    if not use_brute:
        if verbose > 0:
            print("Using KDTree")

        sources_kd = KDTree(sources, 8)
        it = np.nditer(im, flags=["multi_index"])

        while not it.finished:
            loc = it.multi_index
            if verbose > 1:
                print("Processing location (%d, %d)" % (loc[0], loc[1]))

            # Easy way to have len(sources) distinct colors
            im2[loc] = (255 / len(sources)) * sources_kd.query([loc], p=1)[1][0]
            it.iternext()
    else:
        if verbose > 0:
            print("Using brute force")

        # Otherwise, simply use brute force.
        it = np.nditer(im, flags=["multi_index"])

        while not it.finished:
            loc = it.multi_index

            if verbose > 1:
                print("Processing location (%d, %d)" % (loc[0], loc[1]))

            min_dist = float('inf')
            for i in range(len(sources)):
                if cityblock(loc, sources[i]) < min_dist:
                    im2[loc] = (255 / len(sources)) * i
                    min_dist = cityblock(loc, sources[i])

            it.iternext()

    end_time = datetime.now()

    if verbose > 0:
        if use_brute:
            print("Voronoi computation took %d seconds (%d microseconds) using brute force" % ((end_time - start_time).seconds, (end_time - start_time).microseconds))
        else:
            print("Voronoi computation took %d seconds (%d microseconds) using KDTree" % ((end_time - start_time).seconds, (end_time - start_time).microseconds))

    # Plotting
    plt.subplot(1, 2, 1)
    plt.imshow(im, interpolation='nearest', vmin=0, vmax=255)
    plt.title("Source points")

    plt.subplot(1, 2, 2)
    plt.imshow(im2, interpolation='nearest', vmin=0, vmax=255)
    plt.title("Voronoi regions")

    plt.savefig(out_file)
    plt.show()

def prepare_args():
    """
    Method which parses command line arguments to the progam.
    :return: the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate Voronoi regions for randomly sampled points in as specified region")

    parser.add_argument('-s', '--side-length', dest="side_length", help="the side length of the displayed square plot. (Default 100)",
                        type=int, default=100)
    parser.add_argument('-n', '--num-points', dest="num_points", help="the number of random points to generate. (Default 20)",
                        type=int, default=20)

    parser.add_argument('-b', '--brute_force', dest="use_brute", help="whether to use the brute force or KDTree (default) implementation",
                        action='store_true', default=False)

    parser.add_argument('-v', '--verbosity', dest="verbosity", help="which level of debugging information to print. Level 0 \
                                                                prints no information, level 1 prints timing information, \
                                                                and level 2 prints incremental information.",
                        type=int, default=0)

    parser.add_argument('-i', '--image-file', dest="image_file", help="an image file to save the generated plot to (Default out.png)",
                        type=str, default="out.png")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = prepare_args()

    random_voronoi(args.side_length, args.num_points, args.use_brute, args.verbosity, args.image_file)


