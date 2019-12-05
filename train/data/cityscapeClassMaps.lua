-- Eduardo Romera,
-- May 2017.
-- Define Cityscapes class mappings
----------------------------------------------------------------------

classes = {'Unlabeled', 'Road', 'Sidewalk', 'Building', 'Wall', 'Fence',
	             'Pole', 'TrafficLight', 'TrafficSign', 'Vegetation', 'Terrain',
	             'Sky', 'Person', 'Rider', 'Car', 'Truck',
	             'Bus', 'Train', 'Motorcycle', 'Bicycle'}
conClasses = {'Road', 'Sidewalk', 'Building', 'Wall', 'Fence',
	                'Pole', 'TrafficLight', 'TrafficSign', 'Vegetation', 'Terrain',
	                'Sky', 'Person', 'Rider','Car', 'Truck',
	                'Bus', 'Train', 'Motorcycle', 'Bicycle'} -- 19 classes

nClasses = #classes

-- Ignoring unnecessary classes
print '==> remapping classes'

classMap = {  [-1] =  {1}, -- licence plate
                [0]  =  {1}, -- Unlabeled
                [1]  =  {1}, -- Ego vehicle
                [2]  =  {1}, -- Rectification border
                [3]  =  {1}, -- Out of roi
                [4]  =  {1}, -- Static
                [5]  =  {1}, -- Dynamic
                [6]  =  {1}, -- Ground
                [7]  =  {2}, -- Road
                [8]  =  {3}, -- Sidewalk
                [9]  =  {1}, -- Parking
                [10] =  {1}, -- Rail track
                [11] =  {4}, -- Building
                [12] =  {5}, -- Wall
                [13] =  {6}, -- Fence
                [14] =  {1}, -- Guard rail
                [15] =  {1}, -- Bridge
                [16] =  {1}, -- Tunnel
                [17] =  {7}, -- Pole
                [18] =  {1},  -- Polegroup
                [19] =  {8}, -- Traffic light
                [20] =  {9}, -- Traffic sign
                [21] = {10}, -- Vegetation
                [22] = {11}, -- Terrain
                [23] = {12}, -- Sky
                [24] = {13}, -- Person
                [25] = {14}, -- Rider
                [26] = {15}, -- Car
                [27] = {16}, -- Truck
                [28] = {17}, -- Bus
                [29] =  {1}, -- Caravan
                [30] =  {1}, -- Trailer
                [31] = {18}, -- Train
                [32] = {19}, -- Motorcycle
                [33] = {20}, -- Bicycle
                }


if opt.smallNet then
    classMap = {[-1] = {1},  -- licence platete
                [0]  = {1},  -- Unlabeled
                [1]  = {1},  -- Ego vehicle
                [2]  = {1},  -- Rectification border
                [3]  = {1},  -- Out of roi
                [4]  = {1},  -- Static
                [5]  = {1},  -- Dynamic
                [6]  = {1},  -- Ground
                [7]  = {2},  -- Road
                [8]  = {2},  -- Sidewalk
                [9]  = {2},  -- Parking
                [10] = {2},  -- Rail track
                [11] = {3},  -- Building
                [12] = {3},  -- Wall
                [13] = {3}, -- Fence
                [14] = {3}, -- Guard rail
                [15] = {3},  -- Bridge
                [16] = {3},  -- Tunnel
                [17] = {4}, -- Pole
                [18] = {4},  -- Polegroup
                [19] = {4},  -- Traffic Light
                [20] = {4}, -- Traffic iSign
                [21] = {5}, -- Vegetation
                [22] = {5}, -- Terrain
                [23] = {6}, -- Sky
                [24] = {7}, -- Person
                [25] = {7}, -- Rider
                [26] = {8}, -- Car
                [27] = {8}, -- Truck
                [28] = {8}, -- Bus
                [29] = {8}, -- Caravan
                [30] = {8}, -- Trailer
                [31] = {8}, -- Train
                [32] = {8}, -- Motorcycle
                [33] = {8}, -- Bicycle
    }

    classes = {'Unlabeled', 'flat', 'construction', 'object', 'nature',
	           'sky', 'human', 'vehicle', }
    conClasses = {'flat', 'construction', 'object', 'nature','sky', 'human', 'vehicle', } -- 7 classes
end


print('==> number of classes: ' .. #classes)
print('classes are:')
print(classes)
