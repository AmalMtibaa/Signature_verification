import matplotlib.pyplot as plt
import numpy as np
import math

def plot_dataset_random(dataset, n=100, size=128):
    """Plot a random subset of the given dataset
    """
    sample = np.random.randint(0, len(dataset), n)
    plot_dataset(dataset[sample, ])

def plot_dataset(dataset, size=128):
    """Plot the entire given dataset (arranged in a square figure)
    """
    n = math.ceil(math.sqrt(len(dataset)))
    figure = np.zeros((size*n, size*n))
    for i, image in enumerate(dataset):
        if i >= len(dataset):
            break
        x = i // n
        y = i % n
        figure[x*size: (x+1)*size, y*size: (y+1)*size] = image.reshape(size,
                                                                       size)
        plt.figure()
        plt.imshow(figure, cmap='gray')
        plt.show()

def display_one(a, title1="Original"):
    plt.imshow(a, cmap='gray'), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.show()

def display_two(a, b, title1 = "X1", title2 = "X2"):
    plt.subplot(121), plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(b), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.show()

def display_two_with_distance(a, b, distance,title1 = "X1", title2 = "X2"):
    plt.text(15, -0.01, "Distance :" + distance)
    plt.subplot(121), plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(b), plt.title(title2)
    plt.xticks([]), plt.yticks([])

    plt.show()


def plot_distances(x,y,x1,y2):
    plt.plot(x,y,'go')
    plt.plot(x1,y2,'ro')
    plt.xlabel('Distances')
    plt.ylabel('Cos Similarity')
    plt.axvline(x=200)
    plt.axvline(x=350)
    #plt.axis([0,600,0,1])
    #plt.plot(0.78, [y] * len(x))
    plt.show()

