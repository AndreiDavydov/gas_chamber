import numpy as np
import src
import os

RAWVIDEOPATH = os.path.join(src.DATAPATH, "raw/MVI_0022.MP4")

# first frames are shaky; preliminary cut to localize the tube better
NUM_FRAMES_TO_SKIP = 1
BBOX_TO_CUT = (290, 2050, 960, 2720)

# small patch of the frame to align
BBOX_TO_ALIGN = (0, 100, 1600, 1700)

# points to make a mask from frame1 (cut, aligned)
SNAKE_ANCHOR_POINTS = np.array([
    [200, 29],
    [1675, 32],
    [1687, 35],
    [1695, 38],
    [1700, 42],
    [1709, 50],
    [1715, 57],
    [1718, 62],
    [1722, 70],
    [1724, 85],
    #
    [1724, 355],
    [1715, 379],
    [1700, 395],
    [1680, 402],
    [1655, 403],
    #
    [153, 405],
    #
    [153, 550],
    [155,555],
    #
    [1660,557],
    [1725, 625],
    [1720, 880],
    [1680,925],
    #
    [150,925],
    [150,1080],
    [1670,1080],
    [1720,1120],
    [1720,1400],
    [1640,1450],
    [147,1450],
    [150,1600],
    [155,1605],
    #
    [1650,1605],
    #
    #
    #
    [1650,1712],
    [100,1715],
    [42,1660],
    [42,1440],
    [120,1345],
    [1610,1340],
    [1615,1330],
    [1615,1185],
    [100,1185],
    [43,1140],
    [45,860],
    [60,832],
    [80,820],
    #
    [1610, 820],
    [1615,810],
    [1615,662],
    [120,662],
    [80,660],
    [70,655],
    [60,645],
    [50,635],
    [45, 618],
    #
    [47, 375],
    [50, 330],
    [60, 314],
    [70, 304],
    [80, 297],
    [100, 293],
    [153, 293],
    [1615, 292],
    [1617, 138],
    [200, 137],
])