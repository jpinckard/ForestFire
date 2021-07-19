import plotly.graph_objects as go
import plotly.express as px
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
grid_width = 50 # height and width of grid in meters (100 meters x 100 meters)
grid_height= 50 # width of the grid in meters

r_inner = 15
r_outer = 35
a = 0.25
b = 0.05
min_sep = 4
time=20 # simulation length

dist = 3 # distance between cells to check

# Terrain extents defined by south-west and north-east points
# Currently for huntsville area, please change these to a selected forest later
MIN_LON = -86.90567
MIN_LAT = 34.53765
MAX_LON = -86.41380
MAX_LAT = 34.86531


# For converting cell positions to latitude and longitude.
area_width  = MAX_LAT - MIN_LAT
area_height = MAX_LON - MIN_LON

x_scale = grid_width  / area_width
y_scale = grid_height / area_height

# Constants used to calculate the probability of ignition due to elevation and wind.
C1 = float(0.045)
C2 = float(0.131)
A  = float(0.078)

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


class ElevationData:
    def __init__(self, filename):
        global grid_height, grid_width, x_scale, y_scale
        # Get the elevation data from the provided csv file.
        self.df = pd.read_csv(filename)
        # Ensure the grid width and height match the elevation data.
        # For reference, the grid height/width of the csv file is 1200 x 2280.
        grid_height, grid_width = self.df.shape
        x_scale, y_scale = grid_width / area_width, grid_height / area_height

    # Todo: This should likely be defined similarly to get_terrain_altitude.
    def get_elevation_data(self, x, y):
        return self.df.iat[x, y]


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

def Distance(tree1_x, tree1_y, tree2_x, tree2_y):
    dist = math.sqrt((tree2[TREES_X]-tree1[TREES_X])**2 + (tree2[TREES_Y]-tree2[TREES_Y])**2)
    return dist


#############################
## INITIALIZE GRID FOREST ###
# Create the forest as a grid of trees.
def InitGridForest(grid_width, grid_height):

    filename = "forest_fire.csv"
    ed = ElevationData(filename)
    trees = np.empty((grid_width, grid_height, 2))

    for y, row in enumerate (trees):
        for x, col in enumerate (row):
            z = ed.get_elevation_data(x, y)
            trees[y][x] = [z, unburned]


    #z = TerrainAPI.get_terrain_altitude(grid_width/2, grid_height/2)
    x = int(np.floor(grid_width/2))
    y = int(np.floor(grid_height/2))
    trees[y][x] = [0, burning] 

    return trees


############################
## JOY BURN FOREST #############
# Simulates forest fire using SIR model with spatial ignition probability.
def BurnForest(trees, grid_width, grid_height, dist, time):

    x = int(np.floor(grid_width/2))
    y = int(np.floor(grid_height/2))
    
    burning_trees = [[x, y]] # list of indexes of burning trees from list 'trees'. Initially populated with the first burning tree.
    burnt_trees = [[0,0]] # list of indexes of burning trees from list 'trees'. Initially populated with the first burning tree.
    turns = [[0, burnt_trees, burning_trees]] # A record of the changes in tree state for each turn.


    # Burn over time
    for turn in range (time):
        
        burning_trees_len = len(burning_trees)
        new_burnt_trees = []
        
        # Loop through all burning trees
        for j in range(burning_trees_len):

            tree = burning_trees[j]

            # Get burnt out trees
            if (np.random.uniform(0,1,1)[0] < b):                
                trees[tree[1], tree[0]][1] = burnedout
                new_burnt_trees.append(tree)
                burnt_trees.append(tree)

            # Check neighbors for ignition 
            for i in range (-dist, dist+1):
                for k in range (-dist, dist+1):
                    x = tree[0] + i
                    y = tree[1] + k
                    if (x < grid_width and y < grid_height and TryIgnite(trees[y][x])):
                        burning_trees.append([x, y])
                        trees[y][x][1] = burning

        # Remove burning trees from list
        for burnt_tree in new_burnt_trees:
            burning_trees.remove(burnt_tree)
            
        turns.append([turn, burnt_trees, burning_trees])

        DrawGrid(trees)

    return turns
                

############################
## TRY IGNITE ##############
# Checks ignition probability of two trees.
def TryIgnite(tree):
    if (tree[1] == unburned and random.randint(0, 1) == 1):
        return True
    return False

##################
## DRAW GRID #####
def DrawGrid(trees):
    print("draw grid")
    for y, row in enumerate (trees):
        line = ""
        for x, col in enumerate (row):
            line = line + str(int(trees[y][x][1])) + " "
        print(line)
        

##########
## MAIN ##
def Main():
    trees = InitGridForest(grid_width, grid_height)
    burn_forest = BurnForest(trees, grid_width, grid_height, dist, time)
    DrawGrid(trees)
    #output = pd.DataFrame(burn_forest).to_csv("C:/Users/jpinckard/Documents/CS595_Summer2021_MikelPetty/ForestFire/results.csv")


################################################
## LOOP ########################################
Main()
