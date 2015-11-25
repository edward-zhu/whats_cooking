require 'nn'
require 'optim'
require 'pl.text'.format_operator()

local tablex = require 'pl.tablex'

local train_data = torch.load("train.t7")

local hidden = 1024
local gpu = true

if gpu then
    require 'cltorch'
    require 'clnn'
end

local net = torch.load("cooking_model_17.t7")

if gpu then
    net:cl()
end

local params, grad_params = net:getParameters()

-- generating train_set and validate_set

local samples = train_data.train_data
local vals = {}

local n_sample = table.getn(samples)
local n_train = math.floor(n_sample * 0.9)
local n_val = n_sample - n_train


print("total samples: %d, train: %d, validate: %d" % {n_sample, n_train, n_val})

n_sample = n_sample - n_val

tablex.move(vals, samples, 1, n_train, n_val)
tablex.clear(samples, n_train+ 1)


local get_input = function(sample)
    local input = torch.zeros(train_data.ingredient_num)

    tablex.map(function(v)
        input[v] = 1
    end, sample.x)
    
    if gpu then
        input = input:cl()
    end
    
    return input
end

-- testing

net:evaluate()

local right = 0

for i, sample in ipairs(vals) do
    local input = get_input(sample)
    local output = net:forward(input)
    local _, pred = output:max(1)
    pred = pred[1]
    if pred == sample.cuisine then
        right = right + 1
    end
end

print(">> CORRECT %d TOTAL %d ACC %.2f" % {right, n_val, right / n_val})

