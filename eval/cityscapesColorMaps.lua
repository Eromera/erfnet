-- Eduardo Romera,
-- May 2017.
----------------------------------------------------------------------

cityscapesMean = torch.FloatTensor{73.633/255.0, 83.367/255.0, 72.867/255.0}

--this one (classMap) passes from labelIds to trainLabelIds
classMap = {[-1] =  {1}, -- licence plate
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
--From trainIds to LabelIds 
reverseClassMap = {[1] =  {1}, -- ignore
                    [2]  =  {7}, -- Road
                    [3]  =  {8}, -- Sidewalk
                    [4] =  {11}, -- Building
                    [5] =  {12}, -- Wall
                    [6] =  {13}, -- Fence
                    [7] =  {17}, -- Pole
                    [8] =  {19}, -- Traffic light
                    [9] =  {20}, -- Traffic sign
                    [10] = {21}, -- Vegetation
                    [11] = {22}, -- Terrain
                    [12] = {23}, -- Sky
                    [13] = {24}, -- Person
                    [14] = {25}, -- Rider
                    [15] = {26}, -- Car
                    [16] = {27}, -- Truck
                    [17] = {28}, -- Bus
                    [18] = {31}, -- Train
                    [19] = {32}, -- Motorcycle
                    [20] = {33}, -- Bicycle
                    }

--RGB colors per label
trainIdColors = {
                {0,0,0},
                {128,64,128},
                {244,35,232},
                {70,70,70},
                {102,102,156},
                {190,153,153},
                {153,153,153},
                {250,170,30},
                {220,220,0},
                {107,142,35},
                {152,251,152},
                {70,130,180},
                {220, 20, 60},
                {255,  0,  0},
                { 0,  0,142},
                { 0,  0, 70},
                { 0, 60,100},
                { 0, 80,100},
                { 0,  0,230},
                {119, 11, 32}}

