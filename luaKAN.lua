local luaKAN = {}

local math_exp = math.exp
local math_sqrt = math.sqrt
local os_time = os.time
local math_random = math.random

-- Seed the random number generator with the current system time
math.randomseed(os_time())

-- Cubic B-Spline basis functions definitions, precomputed t values can be passed if repetitive calculations are needed
local function b0(t) return (1 - t)^3 / 6 end
local function b1(t) return (3*t^3 - 6*t^2 + 4) / 6 end
local function b2(t) return (-3*t^3 + 3*t^2 + 3*t + 1) / 6 end
local function b3(t) return t^3 / 6 end


-- Sigmoid Linear Unit (SiLU), also known as the Swish function
local function silu(x)
    return x / (1 + math_exp(-x))
end

-- Softmax function
function softmax(x)
    local max = -math.huge
    for i = 1, #x do
        if x[i] > max then
            max = x[i]  -- Find the maximum value to stabilize the computation by reducing numerical overflow
        end
    end

    local sum = 0
    local temp_exp
    for i = 1, #x do
        temp_exp = math_exp(x[i] - max) -- Subtract the max for numerical stability
        x[i] = temp_exp
        sum = sum + temp_exp
    end

    for i = 1, #x do
        x[i] = x[i] / sum  -- Normalize to get probability distribution
    end

    return x
end


-- Function to stabilize the SiLU function and prevent numerical overflows
local function stableSilu(x)
    if x < 0 then
        local ex = math_exp(x)  -- Calculates e^x for negative x to avoid overflow
        return x * ex / (1 + ex)  -- Use e^x in the sigmoid computation
    else
        local ex = math_exp(-x)  -- Calculates e^-x for positive x to avoid overflow
        return x / (1 + ex)  -- Use e^-x in the sigmoid computation
    end
end

-- Compute the value of the j-th cubic B-spline basis function at a given x
local function splineBasis(j, x)
    if j == 1 then
        return b0(x)
    elseif j == 2 then
        return b1(x)
    elseif j == 3 then
        return b2(x)
    elseif j == 4 then
        return b3(x)
    else
        error("Invalid basis function index")
    end
end

-- Evaluate the cubic spline at point x using specified knots and coefficients
local function evaluateSpline(t, c, x)
    local n = #t - 1
    local i = 1
    while i < n and x > t[i+1] do
        i = i + 1
    end
    local localX = (x - t[i]) / (t[i+1] - t[i])
    return b0(localX) * (i-2 >= 1 and c[i-2] or 0) +
           b1(localX) * (i-1 >= 1 and c[i-1] or 0) +
           b2(localX) * (i >= 1 and c[i] or 0) +
           b3(localX) * (i+1 >= 1 and c[i+1] or 0)
end

-- Node class definition
Node = {}
Node.__index = Node

-- Initialization of ADAM parameters inside the Node class constructor
function Node:new(numInputs)
    local o = setmetatable({}, self)
    local limit = 1 / math.sqrt(numInputs)
    o.coeffs = {}        -- Store spline coefficients for each input connection
    o.lastInputs = {}
    o.m = {}             -- First moment vector (mean)
    o.v = {}             -- Second moment vector (uncentered variance)
    o.beta1 = 0.9
    o.beta2 = 0.95
    o.epsilon = 1e-8
    o.t = 0              -- Time step counter
    for i = 1, numInputs do
        o.coeffs[i] = {}
        o.lastInputs[i] = 0  -- Initialize last input storage
        o.m[i] = {}
        o.v[i] = {}
        for j = 1, 4 do
            o.coeffs[i][j] = math.random() * 2 * limit - limit  -- Uniform distribution between [-limit, limit]
            o.m[i][j] = 0
            o.v[i][j] = 0
        end
    end
    return o
end

function Node:evaluate(inputs, knots)
    local output = 0
    for i, input in ipairs(inputs) do
        self.lastInputs[i] = input  -- Store last inputs for gradient calculation
        local splineValue = evaluateSpline(knots, self.coeffs[i], input)
        local siluValue = stableSilu(input)  -- Calculate Stable SiLU for the input
        output = output + splineValue + siluValue  -- Combine spline and SiLU values
    end
    return output
end

function Node:getCoefficients()
    return self.coeffs
end

function Node:updateCoefficients(error, knots, learningRate, lambda)
    for i = 1, #self.lastInputs do
        local localX = (self.lastInputs[i] - knots[1]) / (knots[#knots] - knots[1])
        for j = 1, 4 do
            local splineGrad = error * splineBasis(j, localX)  -- Gradient from spline part
            local siluGrad = error * stableSilu(self.lastInputs[i])  -- Gradient from SiLU part
            local grad = splineGrad + siluGrad  -- Combine gradients
            grad = grad + lambda * self.coeffs[i][j]  -- Include L2 regularization term
            self.coeffs[i][j] = self.coeffs[i][j] - learningRate * grad
        end
    end
end

-- Layer class definition
Layer = {}
Layer.__index = Layer
function Layer:new(numNodes, numInputsPerNode)
    local o = setmetatable({}, self)
    o.nodes = {}
    for i = 1, numNodes do
        o.nodes[i] = Node:new(numInputsPerNode)
    end
    return o
end

function Layer:getNodes()
    return self.nodes
end

function Layer:evaluate(inputs, knots)
    local outputs = {}
    for i, node in ipairs(self.nodes) do
        outputs[i] = node:evaluate(inputs, knots)
    end
    return outputs
end

function Layer:updateCoefficients(error, knots, learningRate)
    for _, node in ipairs(self.nodes) do
        node:updateCoefficients(error, knots, learningRate)
    end
end

-- KANetwork class definition
KANetwork = {}
KANetwork.__index = KANetwork
function KANetwork:new(layerSizes, numInputs)
    local o = setmetatable({}, self)
    o.layers = {}
    local inputs = numInputs
    for i, size in ipairs(layerSizes) do
        o.layers[i] = Layer:new(size, inputs)
        inputs = size
    end
    return o
end

-- Evaluate function for KANetwork with optional softmax application at the output
function KANetwork:evaluate(inputs, knots, applySoftmax)
    local outputs = inputs
    for i, layer in ipairs(self.layers) do
        outputs = layer:evaluate(outputs, knots)
    end

    if applySoftmax and #self.layers > 0 then
        outputs = softmax(outputs)
    end

    return outputs
end


-- ADAM coefficients update function
function Node:updateCoefficientsADAM(error, knots, learningRate)
    self.t = self.t + 1  -- Increment time step
    for i = 1, #self.lastInputs do
        local localX = (self.lastInputs[i] - knots[1]) / (knots[#knots] - knots[1])
        for j = 1, 4 do
            local grad = error * splineBasis(j, localX)
            -- ADAM update rules
            self.m[i][j] = self.beta1 * self.m[i][j] + (1 - self.beta1) * grad
            self.v[i][j] = self.beta2 * self.v[i][j] + (1 - self.beta2) * grad^2
            local m_hat = self.m[i][j] / (1 - self.beta1^self.t)
            local v_hat = self.v[i][j] / (1 - self.beta2^self.t)
            self.coeffs[i][j] = self.coeffs[i][j] - learningRate * m_hat / (math.sqrt(v_hat) + self.epsilon)
        end
    end
end

function KANetwork:train(inputs, outputs, knots, learningRate, epochs, lambda)
    local numCoefficients = self:countCoefficients()  -- Count total coefficients in the network
    for epoch = 1, epochs do
        local totalLoss = 0
        local epochOutputs = {}  -- Store outputs for printing.
        for i, input in ipairs(inputs) do
            local predictedOutputs = self:evaluate(input, knots)
            local predicted = predictedOutputs[#predictedOutputs]  -- Assume single output node at last layer.
            local error = predicted - outputs[i][1]
            local regularization = 0.5 * lambda * self:sumOfSquaredCoefficients() / numCoefficients
            totalLoss = totalLoss + 0.5 * error^2 + regularization  -- Adjusted regularization calculation
            epochOutputs[#epochOutputs + 1] = predicted  -- Collect outputs for this epoch
            for _, layer in ipairs(self.layers) do
                for _, node in ipairs(layer:getNodes()) do
                    node:updateCoefficientsADAM(error, knots, learningRate, lambda)
                end
            end
        end
        -- Print total loss and outputs every 100 epochs
        if epoch % 100 == 0 then
            print(string.format("Epoch: %d, Total Loss: %.4f", epoch, totalLoss))
            print("Outputs at epoch " .. epoch .. ":")
            for i, output in ipairs(epochOutputs) do
                print(string.format("Input: {%d, %d}, Predicted: %.4f, True: %d", inputs[i][1], inputs[i][2], output, outputs[i][1]))
            end
        end
    end
end

function KANetwork:updateCoefficientsADAM(errors, knots, learningRate)
    for layerIndex, layer in ipairs(self.layers) do
        for nodeIndex, node in ipairs(layer.nodes) do
            -- Assume error is specifically aligned to each node output, if not, align it accordingly.
            local specificError = errors[nodeIndex] or 0  -- Getting specific error for each node
            node:updateCoefficientsADAM(specificError, knots, learningRate)
        end
    end
end



function KANetwork:countCoefficients()
    local count = 0
    for _, layer in ipairs(self.layers) do
        for _, node in ipairs(layer:getNodes()) do
            for _, coeffs in ipairs(node:getCoefficients()) do
                count = count + #coeffs
            end
        end
    end
    return count
end

function KANetwork:sumOfSquaredCoefficients()
    local sum = 0
    for _, layer in ipairs(self.layers) do
        for _, node in ipairs(layer:getNodes()) do
            for _, coeffs in ipairs(node:getCoefficients()) do
                for _, coeff in ipairs(coeffs) do
                    sum = sum + coeff^2
                end
            end
        end
    end
    return sum
end

function KANetwork:getLayers()
    return self.layers
end

function printCoefficients(network)
    local layers = network:getLayers()
    for i, layer in ipairs(layers) do
        local nodes = layer:getNodes()
        for j, node in ipairs(nodes) do
            local coeffs = node:getCoefficients()
            local coeffsString = ""
            for _, c in ipairs(coeffs) do
                coeffsString = coeffsString .. "("
                for k, v in ipairs(c) do
                    coeffsString = coeffsString .. string.format("%s%.2f", k > 1 and ", " or "", v)
                end
                coeffsString = coeffsString .. ")"
            end
            print(string.format("L%dN%d %s", i, j, coeffsString))
        end
    end
end

luaKAN.Node = Node
luaKAN.Layer = Layer
luaKAN.KANetwork = KANetwork

return luaKAN