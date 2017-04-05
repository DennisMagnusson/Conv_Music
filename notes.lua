require 'midiparse'
require 'lfs'
require 'optim'
require 'tools'
require 'xlua'
require 'rnn' --TODO Remove, just for a quick test
json = require 'json'

opt = get_args(arg, true)

require 'validate'

if opt.opencl then
	require 'cltorch'
	require 'clnn'
else
	require 'torch'
	require 'nn'
end

data_width = 88
curr_ep = 1
start_ep = 0
start_index = 1

totloss = 0
loss = 0
batches = 0

resume = false
prev_valid = 0

meta = {bs=opt.bs,
		rho=opt.rho,
		ep=opt.ep,
		dl=opt.dl,
		hs=opt.hs,
		drop=opt.drop,
		lr=opt.lr,
		lrd=opt.lrd,
		wd=opt.wd,
		d=opt.d,
		vd=opt.vd,
		opencl=opt.opencl,
		ks=opt.ks
}


function next_batch()
	start_index = start_index + opt.bs

	batch = create_batch(start_index)
	if batch == -1 then
		new_epoch()
		batch = create_batch(start_index)
	end

	batches = batches + 1

	return batch
end

function feval(p)
	if params ~= p then
		params:copy(p)
	end

	batch = next_batch()
	local x = batch[1]

	--Remove these lines to switch to rnn
	x = x:transpose(2, 3)
	x = torch.reshape(x, opt.bs, 1, data_width, opt.rho)
	local y = batch[2]

	gradparams:zero()
	local yhat = model:forward(x)
	local loss = criterion:forward(yhat, y)
	local e = torch.mean(torch.abs(yhat-y))
	totloss = totloss + e--Use real error instead of criterion
	model:backward(x, criterion:backward(yhat, y))

	collectgarbage()

	return loss, gradparams
end

--[[
function create_batch(start_index)
	local i = start_index
	local song = torch.Tensor()
	local songindex = 0
	--Select the correct song
	for k, s in pairs(data) do
		if s:size()[1] > i then
			song = s
			songindex = k
			break
		else
			i = i - s:size()[1]
		end
		if i < 1 then i = 1 end
	end
	--Create batch
	local x = torch.Tensor(opt.bs, opt.rho, data_width)
	local y = torch.Tensor(opt.bs, data_width)

	for u = 1, opt.bs do
		::s::
		if song:size()[1] < i+u+opt.rho+1 then
			song = data[songindex+1]
			if song==nil then return -1 end
			songindex = songindex+1
			i=1
			goto s
		end

		for o = opt.rho, 1, -1 do
			x[u][o] = song[i+o+u]
		end
		y[u] = song[i+u+opt.rho+1]
	end

	if opt.opencl then
		x = x:cl()
		y = y:cl()
	end

	return {x, y}
end
]]

function create_model()
	local model = nn.Sequential()
	
	for n, ks in pairs(opt.ks) do
		local kx = ks[1]
		local ky = ks[2]

		local padx = math.floor(kx/2)
		local pady = math.floor(ky/2)
		local c_in = opt.c[n-1] or 1
		local c_out= opt.c[n]
		model:add(nn.SpatialConvolution(c_in, c_out, kx, ky, 1, 1, padx, pady))
		model:add(nn.ReLU())
	end
	
	size = opt.c[#opt.ks] * data_width * opt.rho
	model:add(nn.Reshape(size))
	model:add(nn.Linear(size, 128))

	--Output layer
	model:add(nn.Dropout(opt.drop))
	model:add(nn.Linear(128, data_width))
	model:add(nn.Sigmoid())
	
	--Let's compare to a recurrent
	--[[
	--TODO RemoveMe
	local rnn = nn.Sequential()
	rnn:add(nn.FastLSTM(88, 256, opt.rho))
	rnn:add(nn.SoftSign())
	rnn:add(nn.FastLSTM(256, 128, opt.rho))
	rnn:add(nn.SoftSign())
	model:add(nn.SplitTable(1, 2))
	model:add(nn.Sequencer(rnn))
	model:add(nn.SelectTable(-1))
	--model:add(nn.Dropout(opt.drop))
	model:add(nn.Linear(128, 128))
	model:add(nn.SoftSign())
	model:add(nn.Linear(128, 88))
	model:add(nn.Sigmoid())
	]]

	if opt.opencl then 
		return model:cl()
	else 
		return model 
	end
end

if lfs.attributes(opt.o) then--Resume training
	model = reload_model(opt.o)
else
	model = create_model()
	
	if opt.o ~= '' then
		logger = optim.Logger(opt.o..".log")
		logger:setNames{'epoch', 'loss', 'delta', 'v_loss', 'v_delta'}
	else print("\n\n\nWARNING: No output file!\n\n\n") end --To prevent future mistakes
end

params, gradparams = model:getParameters()
criterion = nn.BCECriterion()--BCE is waaaay better
if opt.opencl then criterion:cl() end

data = create_dataset(opt.d, false, opt.ds)

totlen = get_total_len(data)

print(curr_ep)
print(start_ep)

train(optim.adam)

if opt.o ~= '' then
	save(model, opt.o)
end
