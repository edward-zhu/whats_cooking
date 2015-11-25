require 'nn'
require 'optim'
require 'pl.text'.format_operator()

local tablex = require 'pl.tablex'

local train_data = torch.load("train_test.t7")

local hidden1 = 1024
local hidden2 = 512
local gpu = true

if gpu then
    require 'cltorch'
    require 'clnn'
end

local net = nn.Sequential()
net:add(nn.Linear(train_data.ingredient_num, hidden1))
net:add(nn.ReLU())
net:add(nn.Dropout(0.5))
net:add(nn.Linear(hidden1, hidden2))
net:add(nn.ReLU())
net:add(nn.Dropout(0.5))
net:add(nn.Linear(hidden2, train_data.cuisine_num))
net:add(nn.LogSoftMax())

local criterion = nn.ClassNLLCriterion()

if gpu then
    net:cl()
    criterion = criterion:cl()
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

-- other

local timer = torch.Timer()

local optim_opts = {
    learningRate = 1e-3,
    momentum = 0.9
}

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

for j = 1, 30 do

    for i, sample in ipairs(samples) do
        local feval = function(params)
            local input = get_input(sample)
            
            local output = net:forward(input)
            local grad, loss
            
            if gpu then
                output = output:reshape(1, train_data.cuisine_num)
                loss = criterion:forward(output, torch.ClTensor{sample.cuisine})
            else
                loss = criterion:forward(output, sample.cuisine)
            end
            
            net:zeroGradParameters()
            
            if gpu then
                grad = criterion:backward(output, torch.ClTensor{sample.cuisine})
                grad = grad:reshape(train_data.cuisine_num)
            else
                grad = criterion:backward(output, sample.cuisine)
            end
            
            output = output:reshape(train_data.cuisine_num)
            
            net:backward(input, grad)
        
            return loss, grad_params
        end
        
        optim.sgd(feval, params, optim_opts)
         
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
    
    print("EPOCH %d\tCORRECT %d\tTOTAL %d\tACC %.6f" % {j, right, n_val, right / n_val})
    
    torch.save("cooking_model_%d.t7" % {j}, net)
    
    net:training()

end

