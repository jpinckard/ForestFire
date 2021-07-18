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
        global grid_height, grid_width
        # Get the elevation data from the provided csv file.
        self.df = pd.read_csv(filename)
        # Ensure the grid width and height match the elevation data.
        # For reference, the grid height/width of the csv file is 1200 x 2280.
        grid_height, grid_width = self.df.shape

    # Todo: This should likely be defined similarly to get_terrain_altitude.
    def get_elevation_data(self, tree):
        x, y = tree[1], tree[0]
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

    #t = TerrainApi()

    for y, row in enumerate (trees):
        for x, col in enumerate (row):
            pos = [(x * x_scale) + MIN_LAT, (y * y_scale) + MIN_LON]
            #z = t.get_terrain_altitude(pos[0], pos[1])
            trees[y][x] = [0, unburned]    


    #z = TerrainAPI.get_terrain_altitude(grid_width/2, grid_height/2)
    x = int(np.floor(grid_width/2))
    y = int(np.floor(grid_height/2))
    trees[y][x] = [0, burning] 

    return trees

############################
## PROBABILITY TO IGNITE ###
# Calculates probability of each tree igniting each other tree.
# This probability based on distance, thus does not change.
def InitPignite(trees, num_trees, a, r_inner, r_outer):

    N = num_trees-1

    p_ignite = np.empty([N, N]) # Creates an NxN array to hold all the tree probs
    
    for i in range (N):
        for j in range (N):
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
        for y, row in enumerate (trees):
            for x, col in enumerate (row):
                if (trees[y][x][1] == burning ):
                    if (np.random.uniform(0,1,1)[0] < b):
                        nextTrees[y][x][1] = burnedout
                        burnedoutCount += 1
                        burningCount -= 1
                elif (trees[y][x][1] == unburned ):
                    Q=1
                    k=0
                    for u, row in enumerate (trees):
                       for w, col in enumerate (row):
                        if (trees[u][w][1] == burning ):
                            Q = Q * (1-p_ignite[u, w])
                    if (np.random.uniform(0,1,1)[0] >= Q):
                        trees[y][x][1] = burning
                        burningCount += 1

        trees = nextTrees
        if (i % 3 == 0):
            DrawFigure(trees)

    #results =[num_trees - burningCount - burnedoutCount, burningCount, burnedoutCount]
    return nextTrees

############################
## JOY BURN FOREST #############
# Simulates forest fire using SIR model with spatial ignition probability.
def BurnForest(trees, num_trees, grid_width, grid_height, dist, time):

    x = int(np.floor(grid_width/2))
    y = int(np.floor(grid_height/2))
    
    burning_trees = [[x, y]] # list of indexes of burning trees from list 'trees'. Initially populated with the first burning tree.
    burnt_trees = [[0,0]] # list of indexes of burning trees from list 'trees'. Initially populated with the first burning tree.
    turns = [[0, burnt_trees, burning_trees]] # A record of the changes in tree state for each turn.

    

    # Burn over time
    for turn in range (time):

        print ("Turn: ", turn)
        print ("Burning trees", burning_trees)
        print ("Length burning trees: ", len(burning_trees))

        burning_trees_len = len(burning_trees)

        if (len(burning_trees) <= 0):
            break



        #print ("Burning trees", burning_trees)
        
        # Loop through all burning trees
        for j in range(burning_trees_len):

            print ("j", j)

            tree = burning_trees[j]

            #print (tree)

            # Get burnt out trees
            if (np.random.uniform(0,1,1)[0] < b):                
                trees[tree[1], tree[0]][1] = burnedout
                burning_trees.remove(tree) # might have to change the syntax of this
                burnt_trees.append(tree)

            # Check neighbors for ignition 
            for i in range (-dist, dist):

                print ("Burning tree num", j)

                # Get neighbors along X and Y axis
                if (i != 0):
                    neighbor_x = [x+i, y]
                    neighbor_y = [x, y+i]
                    neighbor_d = [x+i, y+i] # along the diagonal

                # print ("Tree", trees[neighbor_x[1]][neighbor_x[0]][1] == unburned)

                # Check ignition of those neighbors
                if (TryIgnite(trees[neighbor_x[1]][neighbor_x[0]])):
                    burning_trees.append(neighbor_x)
                    trees[neighbor_x[1]][neighbor_x[0]][1] = burning
                    print("...")
                    
                if (TryIgnite(trees[neighbor_y[1]][neighbor_y[0]])):
                    burning_trees.append(neighbor_y)
                    trees[neighbor_y[1]][neighbor_y[0]][1] = burning
                    
                if (TryIgnite(trees[neighbor_d[1]][neighbor_d[0]])):
                    burning_trees.append(neighbor_d)
                    trees[neighbor_d[1]][neighbor_d[0]][1] = burning

        turns.append([turn, burnt_trees, burning_trees])                
                

############################
## TRY IGNITE ##############
# Checks ignition probability of two trees.
def TryIgnite(tree):
    if (tree[1] == unburned and random.randint(0, 1) == 1):
        print("True This ran")
        return True
    print ("False")
    return False

################################################
## FUNCTIONS ###################################
##########
## MAIN ##
def Main():
    trees = InitGridForest(grid_width, grid_height)
    
    #p_ignite = InitPignite(trees, num_trees, a, r_inner, r_outer)
    burn_forest = BurnForest(trees, num_trees, grid_width, grid_height, dist, time)
    #print(burn_forest)
    #output = pd.DataFrame(burn_forest).to_csv("C:/Users/jpinckard/Documents/CS595_Summer2021_MikelPetty/ForestFire/results.csv")


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
