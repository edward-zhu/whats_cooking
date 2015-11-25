require 'nn'
require 'optim'
require 'pl.text'.format_operator()

local tablex = require 'pl.tablex'
local csvigo = require 'csvigo'

local train_data = torch.load("train_test.t7")

local hidden = 1024
local gpu = true

if gpu then
    require 'cltorch'
    require 'clnn'
end

local net = torch.load("cooking_model_21.t7")

if gpu then
    net:cl()
end

local params, grad_params = net:getParameters()

-- generating train_set and validate_set

local samples = train_data.test_data

print("Got %d tests." % {table.getn(samples)})


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

local submit = {
    id = {},
    cuisine = {}
}

for i, sample in ipairs(samples) do
    local input = get_input(sample)
    local output = net:forward(input)
    local _, pred = output:max(1)
    pred = pred[1]
    table.insert(submit.id, sample.id)
    table.insert(submit.cuisine, train_data.idx_to_cui[pred])
end

csvigo.save{path="submission_1.csv", data=submit}



