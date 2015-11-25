require 'json'
local Set = require 'pl.Set'
local Map = require 'pl.Map'
local tablex = require 'pl.tablex'
local stringx = require 'pl.stringx'

require 'pl.text'.format_operator()

stringx.import()

-- loading smaples
local train_data = json.load("train.json")
local test_data = json.load("test.json")
local ingredient_set = Set()
local cuisine_set = Set()

local gen_ingredients = function(ing_str)
    ing_str = string.lower(ing_str)
    return ing_str:split(" ")
end

print("Loading train set ingredients & cuisines...")
tablex.map(function(v)
    local ing_str = (" "):join(v.ingredients)
    v.ingredients = gen_ingredients(ing_str)
    ingredient_set = ingredient_set + Set(v.ingredients)
    cuisine_set = cuisine_set + Set({v.cuisine})
end, train_data)

print("Loading test set ingredients & cuisines...")

tablex.map(function(v)
    local ing_str = (" "):join(v.ingredients)
    v.ingredients =  gen_ingredients(ing_str)
    ingredient_set = ingredient_set + Set(v.ingredients)
    cuisine_set = cuisine_set + Set({v.cuisine})
end, test_data)

print("generating codec...")
local idx_to_cui = Set.values(cuisine_set)
local cui_to_idx = Map()
local idx_to_ing = Set.values(ingredient_set)
local ing_to_idx = Map()

local ingredient_num = Set.len(ingredient_set)
local cuisine_num = Set.len(cuisine_set)

print("Got %d ingredients, %d cuisines." % {ingredient_num, cuisine_num})

for i, x in ipairs(idx_to_ing) do
    ing_to_idx[x] = i
end

for i, x in ipairs(idx_to_cui) do
    cui_to_idx[x] = i
end

print("Generating train_data...")

tablex.map(function(v)
    v.x = {}
    tablex.map(function(item)
        table.insert(v.x, ing_to_idx[item])
    end, v.ingredients)
    v.cuisine = cui_to_idx[v.cuisine]
    v.ingredients = nil
end, train_data)

print("Generating test_data...")

tablex.map(function(v)
    v.x = {}
    tablex.map(function(item)
        table.insert(v.x, ing_to_idx[item])
    end, v.ingredients)
    v.cuisine = ""
    v.ingredients = nil
end, test_data)

torch.save("train_test.t7", {
    idx_to_cui = idx_to_cui,
    cui_to_idx = cui_to_idx,
    idx_to_ing = idx_to_ing,
    ing_to_idx = ing_to_idx,
    train_data = train_data,
    test_data = test_data,
    cuisine_num = cuisine_num,
    ingredient_num = ingredient_num
})





