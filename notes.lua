require 'midiparse'
require 'lfs'
require 'optim'
require 'tools'
require 'xlua'
require 'rnn' --TODO Remove, just for a quick test
json = require 'json'

cmd = torch.CmdLine()
cmd:option('-d', 'slowtest', 'Dataset directory')
cmd:option('-vd', '', 'Validation data directory')
cmd:option('-datasize', 0, 'Size of dataset (for benchmarking)')
cmd:option('-o', '', 'Model filename')
cmd:option('-ep', 1, 'Number of epochs')
cmd:option('-batchsize', 128, 'Batch Size')
cmd:option('-rho', 16, 'Rho value')
cmd:option('-denselayers', 1, 'Number of dense layers')--TODO Do I need multiple dense layers?
cmd:option('-channels', '2,5', 'Sizes of hidden layers, seperated by commas')
cmd:option('-kernelsizes', '13x5,5x5', 'Kernel sizes, following the same form as default value')--TODO
cmd:option('-dropout', 0.5, 'Dropout probability')
cmd:option('-lr', 0.0001, 'Learning rate')
cmd:option('-lrdecay', 1e-5, 'Learning rate decay')
cmd:option('-cpu', false, 'Use CPU')
cmd:option('-weightdecay', 0, 'Weight decay')
opt = cmd:parse(arg or {})

opt.opencl = not opt.cpu

require 'validate'

if opt.opencl then
	require 'cltorch'
	require 'clnn'
else
	require 'torch'
	require 'nn'
end

--Parse channels
local h = opt.channels
opt.channels = {}
while true do
	if h:len() == 0 then break end
	local c = h:find(',') or h:len()+1
	local str = h:sub(1, c-1)
	h = h:sub(c+1, h:len())
	opt.channels[#opt.channels+1] = tonumber(str)
end

--Parse kernel sizes
h = opt.kernelsizes
opt.kernelsizes = {}
while true do
	if h:len() == 0 then break end
	local c = h:find(',') or h:len()+1
	local str = h:sub(1, c-1) --Selects part between commas
	local x_index = h:find('x')
	local n1 = h:sub(1, x_index-1)
	local n2 = h:sub(x_index+1, c-1)--Right before the next comma
	opt.kernelsizes[#opt.kernelsizes+1] = {tonumber(n1), tonumber(n2)}
	h = h:sub(c+1, h:len())
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

meta = {batchsize=opt.batchsize,
		rho=opt.rho,
		ep=opt.ep,
		denselayers=opt.denselayers,
		hiddensizes=opt.hiddensizes,
		dropout=opt.dropout,
		lr=opt.lr,
		lrdecay=opt.lrdecay,
		weightdecay=opt.weightdecay,
		d=opt.d,
		vd=opt.vd,
		opencl=opt.opencl,
		kernelsizes=opt.kernelsizes
}

function new_epoch()
	start_index = 1
	local prev_loss = loss
	loss = totloss/batches
	local delta = loss-prev_loss
	model:evaluate()
	validation_err = validate(model, opt.rho, opt.batchsize, opt.vd, criterion)
	model:training()
	local v_delta = validation_err - prev_valid
	prev_valid = validation_err

	print(string.format("Ep %d loss=%.8f  dl=%.6e  valid=%.8f  dv=%.6e", curr_ep, loss, delta, validation_err, v_delta))
	if logger then
		logger:add{curr_ep, loss, delta, validation_err, v_delta}
	end

	curr_ep=curr_ep+1

	if(curr_ep % 10 == 0 and opt.o ~= '') then torch.save(opt.o, model) end --Autosave
	collectgarbage()
	totloss = 0
	batches = 0
end

function next_batch()
	start_index = start_index + opt.batchsize

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
	x = torch.reshape(x, opt.batchsize, 1, data_width, opt.rho)
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

function train()
	model:training()--Training mode
	math.randomseed(os.time())

	local optim_cfg = {learningRate=opt.lr, learningRateDecay=opt.lrdecay, weightDecay=opt.weightdecay}
	local progress = -1

	for e=1, opt.ep do
		while curr_ep == start_ep+e do
			if progress ~= math.floor(100*(start_index/totlen)) then
				progress = math.floor(100*(start_index/totlen))
				xlua.progress(100*(e-1)+progress, 100*opt.ep)
			end

			optim.adam(feval, params, optim_cfg)
			collectgarbage()
		end
	end

	model:evaluate() --Exit training mode
end

function get_total_len(data)
	local i = 0
	for k, s in pairs(data) do
		i = i + s:size()[1]
	end
	return i
end

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
	local x = torch.Tensor(opt.batchsize, opt.rho, data_width)
	local y = torch.Tensor(opt.batchsize, data_width)

	for u = 1, opt.batchsize do
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

function create_model()
	local model = nn.Sequential()
	
	for n, kernelsize in pairs(opt.kernelsizes) do
		local kx, ky = kernelsize
		local padx = math.floor(kx/2)
		local pady = math.floor(ky/2)
		local c_in = opt.channels[n-1] or 1
		local c_out= opt.channels[n]
		model:add(nn.SpatialConvolution(c_in, c_out, kx, ky, 1, 1, padx, pady))
		model:add(nn.ReLU())
		--TODO Dropout? Prob not
	end
	
	size = opt.channels[#opt.kernelsizes] * data_width * opt.rho
	model:add(nn.Reshape(size))
	model:add(nn.Linear(size, 128))

	--Output layer
	model:add(nn.Dropout(opt.dropout))
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
	--model:add(nn.Dropout(opt.dropout))
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
	model = torch.load(opt.o)
	resume = true
	--Read JSON
	local file = assert(io.open(opt.o..".meta", 'r'))
	meta = json.decode(file:read('*all'))
	file:close()
	filename = opt.o
	ep = opt.ep

	--Copy table
	for key, val in pairs(meta) do
		opt[key] = val
	end

	opt.o = filename
	opt.ep = ep
	opt.datasize = 0

	print(opt)

	curr_ep = meta['ep']+1
	start_ep = meta['ep']
	opt.lr = meta['lr']/(1+meta['lrdecay']*meta['ep'])--Restore decayed lr
	meta['ep'] = meta['ep'] + opt.ep
	print("opt.ep:", opt.ep, "meta.ep", meta.ep)
	logger = optim.Logger(opt.o..".log2")
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

data = create_dataset(opt.d, false, opt.datasize)

if opt.datasize ~= 0 then
	local l = #data
	for i=opt.datasize, l do
		data[i] = nil
	end
end

totlen = get_total_len(data)

print(curr_ep)
print(start_ep)

train()

if opt.o ~= '' then
	torch.save(opt.o, model)
	local file = assert(io.open(opt.o..".meta", 'w'))
	file:write(json.encode(meta))
	file:close()
	--Merge the logs
	if resume then os.execute("cat "..opt.o..".log2 >> "..opt.o..".log") end
end
