local luaKAN = require "luaKAN"

-- Example XOR problem configuration
local inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}}
local outputs = {{0}, {1}, {1}, {0}}
local knots = {0.2, 0.5}  -- Define the knot vector for splines
local myNetwork = KANetwork:new({3, 2}, 2)  -- A simple 2-layer network suitable for XOR

-- Parameters for training
local learningRate = 0.000015
local epochs = 100000
local trainLambda = 1

-- Train the network
myNetwork:train(inputs, outputs, knots, learningRate, epochs, trainLambda)
printCoefficients(myNetwork)