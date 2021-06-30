import plotly.graph_objects as go
import random
import animation
import numpy

################################################
## VARIABLES ###################################
num_rounds = 1 # number of rounds in the simulation
grid_size = 1000 # height and width of grid in meters (100 meters x 100 meters)
num_trees = 1600 # number of trees in model
num_neighbors = 3 # number of trees to change state

frames = [] # Frames of animation. A collection of grid info.

trees  = {} # x, y coordinate of each tree

############
## COLORS ##
green = "Green"
red   = "Red"
black = "Black"

unburned  = 3
burning   = 2
burnedout = 1

# FIXME
unburned  = "Green"
burning   = "Red"
burnedout = "Black"

#####################
## GENERATE TREES ###
# Generate tree x and y positions on startup
def GenerateTrees(grid_size, num_trees):

    # Generate a forest of trees at random positions.
    #trees_x = [random.randrange(0, grid_size) for i in range(num_trees)]
    #trees_y = [random.randrange(0, grid_size) for i in range(num_trees)]
    trees_x = numpy.random.uniform(0,grid_size,num_trees)
    trees_y = numpy.random.uniform(0,grid_size,num_trees)
    states  = [unburned for i in range (num_trees)]

    trees_x[1] = grid_size/2
    trees_y[1] = grid_size/2
    states[1]  = burning

    trees = {"trees_x": trees_x, "trees_y": trees_y, "state":states}#states} # all trees are initially unburned

    return trees


################################################
## FUNCTIONS ###################################
##########
## MAIN ##
def Main():
    trees = GenerateTrees(grid_size, num_trees)

    for i in range (0, num_rounds):
        ## DRAW UPDATES
        DrawFigure(trees)
        frames.append(go.Frame(data=[go.Scatter(x=[3, 4], y=[3, 4])], layout=go.Layout(title_text="End Title")))

    animate(frames)


##################
## DRAW FIGURE ###
def DrawFigure(trees):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=trees["trees_x"], y=trees["trees_y"],
        name="Forest Fire Simulation",
        mode="markers",
        marker_color=trees["state"]
    ))
    
    # Set options common to all traces with fig.update_traces
    #fig.update_traces(mode='markers', marker_line_width=0, marker_size=10)
    #fig.update_layout(title='Forest Fire Simulation',
    #                  yaxis_zeroline=False,
    #                  xaxis_zeroline=False)
    
    fig.show()

#######################
## GET MARKER COLOR ###
def GetMarkerColor(state):
    # print("state:" + state)
    if   (state == unburned):
        return green
    elif (state == burning):
        return red
    elif (state == burnedout):
        return black
    return "Blue"

#####################
## GET ANIM FRAME ###
def GetFrame():
    return go.Frame(
        data=[
            go.Scatter(
            x=trees["trees_x"], y=trees["trees_y"],
            name="Forest Fire Simulation",
            mode="markers",
            marker_color=GetMarkerColor(trees["state"]))
        ]
    )


################
## SET TREES ###
# Update tree state values.
def SetTrees(trees):
    trees = {"trees_x": trees.get("trees_x"), "trees_y": trees.get("trees_y"), "state":trees.get("state")} # all trees are initially green

########################
## CHANGE TREE STATE ###
# Change the state of a tree to either red or black
def ChangeState(state):
    if (state == green):
        return red
    return black


#######################
## NEIGHBORS ##
# Returns a list of n trees closest to the target tree.
# tree = the tree to target
# d = distance
# FIXME
def GetNeighbors(tree, trees, d):
    neighbors = []

    #for i in range (num_trees):
     #   dist = GetDistance([trees[]])
        ##
        # FOR TESTING PURPOSES. FIX ME
        #randomTree = random.randint(0, len(trees.get("state")))
        ##
  
        #trees.get("state")[randomTree] = ChangeState(trees.get("state")[randomTree])

        
        
      #  neighbors.append(tree)

    # return neighbors

##################
## GET DISTANCE ##
# Returns the distance between a pair of X,Y coordinates
def GetDistance (tree1, tree2):
    X1 = tree1[0]
    X2 = tree2[0]
    Y1 = tree1[1]
    Y2 = tree2[1]
    return(math.sqrt((X2 - X1)^2 + (Y2 - Y1)^2))


################################################
## LOOP ########################################
Main()
