import random
import math
import numpy as np
import pandas as pd

################################################
# VARIABLES ###################################
r_inner = 15
r_outer = 35
a = 0.25
b = 0.05
min_sep = 4
time = 20  # simulation length
cell_distance = 3  # distance between cells to check

# Terrain extents defined by south-west and north-east points
# Currently for huntsville area, please change these to a selected forest later
MIN_LON = -86.90567
MIN_LAT = 34.53765
MAX_LON = -86.41380
MAX_LAT = 34.86531

# Wind speed and direction.
WIND_SPEED = 10  # m/s
WIND_DIRECTION = 3 * np.pi / 2  # Radians, travelling westward by default

# For converting cell positions to latitude and longitude.
grid_width = 1200  # height and width of grid in meters (100 meters x 100 meters)
grid_height = 1200  # width of the grid in meters

area_width = MAX_LON - MIN_LON
area_height = MAX_LAT - MIN_LAT

x_scale = area_width / grid_width
y_scale = area_height / grid_height

# Constants used to calculate the probability of ignition due to elevation and wind.
C1 = float(0.045)
C2 = float(0.131)
A = float(0.078)

trees = []  # x, y coordinate of each tree
filename = "forest_fire.csv"

colors = ["Black", "Red", "Green"]

UNBURNED = 0
BURNING = 1
BURNT = 2

TREES_X = 0  # Position of x variable in trees array
TREES_Y = 1  # position of y variable in trees array
TREES_STATE = 2  # position of states variable in trees array


class ElevationData:
    def __init__(self):
        global grid_height, grid_width, x_scale, y_scale
        # Get the elevation data from the provided csv file.
        self.df = pd.read_csv(filename)
        # Ensure the grid width and height match the elevation data.
        # For reference, the grid height/width of the csv file is 1200 x 2280.
        grid_height, grid_width = self.df.shape
        x_scale, y_scale = area_width / grid_width, area_height / grid_height

    def get_altitude(self, x, y):
        """
        :param x: x-coordinate of grid
        :param y: y-coordinate of grid
        :return: altitude of xy-coordinate
        """
        # Todo: Sometimes throws an out-of-range error.
        return self.df.iat[y, x]


def probability_wind(fire_angle):
    """
    :param fire_angle: direction fire is traveling (angle between burning and unburned tress) in radians
    :return: probability of ignition due to wind
    """
    #print("fire angle is", fire_angle)
    return np.exp(C1 * WIND_SPEED) * np.exp(WIND_SPEED * C2 * (np.cos(WIND_DIRECTION - fire_angle) - 1))


def probability_elevation(e1, e2, distance):
    """
    :param e1: height of the burning tree in meters
    :param e2: height of the unburned tree in meters
    :param distance: distance between the trees in meters
    :return: probability of ignition due to elevation
    """
    if (distance > 0):
        return np.exp(A * np.arctan((e1 - e2) / distance))
    else:
        return 0


def get_geo_pos(x, y):
    """
    :param x: x-coordinate of grid
    :param y: y-coordinate of grid
    :return: longitude-latitude coordinate pair
    """
    return x * x_scale + MIN_LON, y * y_scale + MIN_LAT


# probability of ignition due to distance
def get_distance(tree1_x, tree1_y, tree2_x, tree2_y):
    x1, y1 = get_geo_pos(tree1_x, tree1_y)
    x2, y2 = get_geo_pos(tree2_x, tree2_y)
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


############################
# TRY IGNITE ##############
# Checks ignition probability of two trees.
# Todo: Pass in cell distance and enhance p_ignite.
def try_ignite(burning_pos, unburned_pos):
    # and random.randint(0, 1) == 1):
    burning_x, burning_y, burning_alt = burning_pos
    unburned_x, unburned_y, unburned_alt = unburned_pos

    # Slope is used to calculate angle. The slope at x = 0 is undefined.
    try:
        slope = (unburned_y - burning_y) / (unburned_x - burning_x)
        fire_angle = np.arctan(slope)

    except ZeroDivisionError:
        if unburned_y < burning_y:
            fire_angle = -np.pi / 2
        else:
            fire_angle = np.pi / 2

    distance = get_distance(unburned_x, burning_x, unburned_y, burning_y)

    # Todo: Both probabilities are sometimes over one -- should they be normalized?
    p_wind = probability_wind(fire_angle)
    # print("p_wind is {}".format(p_wind))
    p_elev = probability_elevation(burning_alt, unburned_alt, distance)
    # print("p_elevation is {}".format(p_elev))
    p_ignite = p_elev * p_wind * a

    return random.uniform(0, 1) < p_ignite


############################
# JOY BURN FOREST #############
# Simulates forest fire using SIR model with spatial ignition probability.
def burn_forest(ed):
    global trees
    first_x, first_y = grid_width // 2, grid_height // 2

    burning_trees = [[first_x, first_y]]  # Burning trees from 'trees'; initially populated with the first burning tree.
    burnt_trees = [[0, 0]]  # Burnt trees from 'trees'; initially populated with the first burnt tree.
    tree_records = [[0, first_x, first_y, BURNING]]  # Records of the changes in tree state for each turn.

    # Burn over time
    for turn in range(time):
        #print("Turn {}:".format(turn))
        # Loop through all burning trees
        for i  in range (len(burning_trees)):
            burning_tree = burning_trees[i]
            burning_x, burning_y = burning_tree[0], burning_tree[1]
            burning_pos = (burning_x, burning_y, ed.get_altitude(burning_x, burning_y))
            # Did tree burn out?
            if np.random.uniform(0, 1) < b:
                trees[burning_y][burning_x] = BURNT
                burnt_trees.append(burning_tree)
                burning_trees.remove(burning_tree)
                tree_records.append([turn, burning_x, burning_y, BURNT])
                #print("tree burnt out at ({}, {})".format(burning_x, burning_y))
            # Check neighbors for ignition
            # Todo: As cell_distance is greater in magnitude, probability of ignition decreases.
            for i in range(-cell_distance, cell_distance + 1):
                for j in range(-cell_distance, cell_distance + 1):
                    unburned_x, unburned_y = burning_x + i, burning_y + j
                    unburned_pos = (unburned_x, unburned_y, ed.get_altitude(unburned_x, unburned_y))
                    if unburned_x < grid_height and unburned_y < grid_width and \
                            trees[unburned_y][unburned_x] == UNBURNED and \
                            try_ignite(burning_pos, unburned_pos):
                        trees[unburned_y][unburned_x] = BURNING
                        burning_trees.append([unburned_x, unburned_y])
                        tree_records.append([turn, unburned_x, unburned_y, BURNING])

                        #print("burnt tree at ({}, {})".format(unburned_x, unburned_y))

    # Return records for later output.
    return tree_records


##################
# DRAW GRID #####
def draw_grid(trees):
    print("Next Frame")
    for y, row in enumerate(trees):
        line = ""
        for x, col in enumerate(row):
            line = line + str(int(trees[y][x])) + " "
        print(line)


##########
# MAIN ##
def main():
    global trees

    ed = ElevationData()
    trees = np.zeros((grid_width, grid_height))
    trees = burn_forest(ed)
    #draw_grid(trees)
    np.savetxt("tree_output.csv", trees, fmt="%s", delimiter=",", header="Turn,  Tree X, Tree Y, State")
    # pd.DataFrame(burn_forest).to_csv("results.csv")


################################################
# LOOP ########################################
#if __name__ == "__main__":
main()
