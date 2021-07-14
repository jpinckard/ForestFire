import plotly.graph_objects as go
import random
import animation
import math
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import elevation
import richdem as rd
import pandas as pd


################################################
## VARIABLES ###################################
num_rounds = 1    # number of rounds in the simulation
grid_width = 1000  # height and width of grid in meters (100 meters x 100 meters)
grid_height= 1000 # width of the grid in meters
num_trees = 100  # number of trees in model
num_neighbors = 3 # number of trees to change state

r_inner = 15
r_outer = 35
a = 0.25
b = 0.05
min_sep = 4
time=120 # simulation length

# Terrain extents defined by south-west and north-east points
# Currently for huntsville area, please change these to a selected forest later
MIN_LON = -86.90567
MIN_LAT = 34.53765
MAX_LON = -86.41380
MAX_LAT = 34.86531

# Constants used to calculate the probability of ignition due to elevation and wind.
C1 = float(0.045)
C2 = float(0.131)
A  = float(0.078)


frames   = [] # Frames of animation. A collection of grid info.
trees    = [] # x, y coordinate of each tree
p_ignite = []

colors = ["Black", "Red", "Green"]
unburned  = 2
burning   = 1
burnedout = 0


TREES_X = 0 # Position of x variable in trees array
TREES_Y = 1 # position of y variable in trees array
TREES_STATE  = 2 # position of states variable in trees array

# Class used to handle any terrain requests
class TerrainApi:
    def __init__(self):
        # download the terrain elevation data and clip to the specified boundaries
        terrain_path = os.path.join(os.getcwd(), 'test-DEM.tif')
        elevation.clip(bounds=(MIN_LON, MIN_LAT, MAX_LON, MAX_LAT), output=terrain_path)
        self.terrain_data = rd.LoadGDAL(terrain_path)
        # cache the sizes of the elevation database
        self.lat_len = len(self.terrain_data)
        self.lon_len = len(self.terrain_data[0])

    # Get the height above sea level for the specified lat/lon
    def get_terrain_altitude(self, latitude, longitude):
        # find the correct index storing the elevation data
        z1 = (latitude - MIN_LAT) / (MAX_LAT - MIN_LAT) * self.lat_len
        z2 = (longitude - MIN_LON) / (MAX_LON - MIN_LON) * self.lon_len
        # if time permits this should be changed to interpolate between points
        return self.terrain_data[int(z1)][int(z2)]


# probability of ignition due to wind
# speed is wind speed in m/s
# wind_angle is direction of wind in radians
# fire_angle is direction fire is traveling (angle between burning and unburned tress) in radians
def probability_wind(speed, wind_angle, fire_angle):
    return np.exp(C1 * speed) * np.exp(speed * C2 * (np.cos(wind_angle - fire_angle) - 1))


# probability of ignition  due to elevation
# e1 is height of the burning tree, e2 is the height of the unburned tree in meters
# dist is the distance between the trees in meters
def probability_elevation(e1, e2, dist):
    return np.exp(A * np.arctan((e1 - e2)/dist))

def Distance(tree1, tree2):

    dist = math.sqrt((tree2[TREES_X]-tree1[TREES_X])**2 + (tree2[TREES_Y]-tree2[TREES_Y])**2)
    return dist


#############################
## INITIALIZE GRID FOREST ###
# Create the forest as a grid of trees.
def InitGridForest(grid_width, grid_height):
    
    trees = np.empty((grid_width, grid_height, 2))

    area_width  = MAX_LAT - MIN_LAT
    area_height = MAX_LON - MIN_LON

    x_scale = grid_width  / area_width
    y_scale = grid_height / area_height

    t = TerrainApi()

    for y, row in enumerate (trees):
        for x, col in enumerate (row):
            pos = [(x * x_scale) + MIN_LAT, (y * y_scale) + MIN_LON]
            trees[y][x] = [t.get_terrain_altitude(0, pos[0], pos[1]), unburned]    


    trees[grid_width/2][grid_height/2] = [TerrainAPI.get_terrain_altitude(grid_width/2, grid_height/2), burning]    

    return trees


########################
## INITIALIZE FOREST ###
# Creates initial forest randomly located in 2D rectangle.
# No two trees are less than MinDistance apart.
def InitForest(grid_height, grid_width, num_trees, min_sep):

    # Generate a forest of trees at random positions.
    trees = np.array([[grid_width/2, grid_height/2, burning]]) # all trees are initially unburned

    i=0
    while (len(trees) < num_trees):

        tree_x = np.random.uniform(0,grid_width,1)[0]
        tree_y = np.random.uniform(0,grid_height,1)[0]
        state  = unburned

        tree = [tree_x, tree_y, state]

        tooclose = False;

        i = 0
        n = len(trees)
        while i < n-1:
          distance = Distance(tree, trees[i])
          #print("Checking!")
          if (distance < min_sep):
              tooclose = True;
              break;
          i += 1

        if (tooclose == False):
            #print(tree)
            trees = np.vstack([trees, tree])
         
    # Create initial burning tree
    trees = np.vstack([trees, [grid_width/2, grid_height/2, burning]])
    
    return trees

#############################
## O(n) INITIALIZE FOREST ###
## n = 500
## sensitivity = 0.8 # 0 means no movement, 1 means max distance is init_dist
def InitPointForest(n, sensitivity):

    shape = np.array([grid_width, grid_height])
    
    # compute grid shape based on number of points
    width_ratio = shape[1] / shape[0]
    num_y = np.int32(np.sqrt(n / width_ratio)) + 1
    num_x = np.int32(n / num_y) + 1

    # create regularly spaced neurons
    x = np.linspace(0., shape[1]-1, num_x, dtype=np.float32)
    y = np.linspace(0., shape[0]-1, num_y, dtype=np.float32)
    coords = np.stack(np.meshgrid(x, y), -1).reshape(-1,2)

    # compute spacing
    init_dist = np.min((x[1]-x[0], y[1]-y[0]))
    min_dist = init_dist * (1 - sensitivity)

    assert init_dist >= min_dist
    print(min_dist)

    # perturb points
    max_movement = (init_dist - min_dist)/2
    noise = np.random.uniform(
        low=-max_movement,
        high=max_movement,
        size=(len(coords), 2))
    coords += noise

    plt.figure(figsize=(10*width_ratio,10))
    plt.scatter(coords[:,0], coords[:,1], s=3)
    plt.show()

    return shape

############################
## PROBABILITY TO IGNITE ###
# Calculates probability of each tree igniting each other tree.
# This probability based on distance, thus does not change.
def InitPignite(trees, num_trees, a, r_inner, r_outer):

    N = num_trees-1

    p_ignite = np.empty([N, N]) # Creates an NxN array to hold all the tree probs
    
    for i in range (0, N):
        for j in range (0, N):
            dist_ij = Distance(trees[i], trees[j])
            if (dist_ij >= r_outer):
                p_ignite[i, j] = 0
            elif (dist_ij <= r_inner):
                p_ignite[i, j] = a
            else:
                p_ignite[i, j] = a * (1 - ((dist_ij - r_inner)/(r_outer - r_inner) ))

    return p_ignite

############################
## BURN FOREST #############
# Simulates forest fire using SIR model with spatial ignition probability.
def BurnForest(trees, num_trees, grid_width, grid_height, time, a, b, p_ignite, trial_num):
    i=0
    burningCount   = 0
    burnedoutCount = 0
    
    for i in range (time):
        nextTrees = trees
        j=0
        for j in range (num_trees-1):
            if (trees[j][TREES_STATE] == burning ):
                if (np.random.uniform(0,1,1)[0] < b):
                    nextTrees[j][TREES_STATE] = burnedout
                    burnedoutCount += 1
                    burningCount -= 1
            elif (trees[j][TREES_STATE] == unburned ):
                Q=1
                k=0
                for k in range (num_trees-1):
                    if (trees[k][TREES_STATE] == burning ):
                        Q = Q * (1-p_ignite[j,k])
                if (np.random.uniform(0,1,1)[0] >= Q):
                    nextTrees[j][TREES_STATE] = burning
                    burningCount += 1

        trees = nextTrees
        if (i % 3 == 0):
            DrawFigure(trees)

    #results =[num_trees - burningCount - burnedoutCount, burningCount, burnedoutCount]
    return nextTrees


################################################
## FUNCTIONS ###################################
##########
## MAIN ##
def Main():
    #trees = InitPointForest(num_trees, 0.8)
    #trees = InitForest(grid_height, grid_width, num_trees, min_sep)
    trees = InitGridForest(grid_width, grid_height)
    p_ignite = InitPignite(trees, num_trees, a, r_inner, r_outer)
    burn_forest = BurnForest(trees, num_trees, grid_width, grid_height, time, a, b, p_ignite, 1)
    print(burn_forest)
    output = pd.DataFrame(burn_forest).to_csv("C:/Users/jpinckard/Documents/CS595_Summer2021_MikelPetty/ForestFire/results.csv")
    trees = InitForest(num_trees, 0.8)


##################
## DRAW FIGURE ###
def DrawFigure(trees):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=trees[0:num_trees, TREES_X], y=trees[0:num_trees, TREES_Y],
        name="Forest Fire Simulation",
        mode="markers",
        marker_color=trees[0:num_trees, TREES_STATE]
    ))
    
    fig.show()


################################################
## LOOP ########################################
Main()
