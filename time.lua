require 'midiparse'
require 'lfs'
require 'optim'
require 'tools'
require 'xlua'
require 'rnn'
json = require 'json'

opt = get_args(arg, false)

require 'validate'

if opt.opencl then
	require 'cltorch'
	require 'clnn'
else
	require 'torch'
	require 'nn'
end

data_width = 89
curr_ep = 1
start_ep = 0
start_index = 1

totloss = 0
loss = 0
batches = 0 --TODO Can be constant, probably somehow

resume = false
prev_valid = 0

meta = {bs=opt.bs,
		rho=opt.rho,
		ep=opt.ep,
		dl=opt.dl,
		rl=opt.rl,
		hs=opt.hs,
		drop=opt.drop,
		lr=opt.lr,
		lrd=opt.lrd,
		wd=opt.wd,
		d=opt.d,
		vd=opt.vd,
		opencl=opt.opencl
}

-- Min-Maxed logarithms for data with long tail
-- x_n = (ln(x+1)-ln(x_min)) / (ln(x_max)-ln(m_min))
function normalize_col(r, col)
	local min = 99990
	local max = 0
	for i=1, #r do
		for u=1, r[i]:size()[1] do
			--Clamp max dt to 4s
			if r[i][u][col] > 4000 then r[i][u][col] = 4000 end
			r[i][u][col] = math.log(r[i][u][col]+1)-- +1 to prevent ln(0)
			local val = r[i][u][col]
			if min > val then min = val end
			if max < val then max = val end
		end
	end

	for i=1, #r do
		for u=1, r[i]:size()[1] do
			r[i][u][col] = (r[i][u][col] - min)/math.log(4000)
		end
	end

	meta[col..'min'] = min
	meta[col..'max'] = max

	return r
end

function next_batch(time)
	batch = create_batch(data, start_index, true)
	if batch == -1 then
		new_epoch(time)
		batch = create_batch(data, start_index, true)
	end

	start_index = start_index + opt.bs

	batches = batches + 1

	return batch
end

function feval(p)
	if params ~= p then
		params:copy(p)
	end

	batch = next_batch(true)
	local x = batch[1]
	local y = batch[2]

	gradparams:zero()
	local yhat = model:forward(x)
	local train_loss = criterion:forward(yhat, y)
	local e = torch.mean(torch.abs(yhat-y))
	totloss = totloss + e--Use real error instead of criterion
	model:backward(x, criterion:backward(yhat, y))

	collectgarbage()

	return train_loss, gradparams
end

function create_model()
	local model = nn.Sequential()
	local rnn = nn.Sequential()

	local l = 1
	
	--Recurrent layer
	rnn:add(nn.FastLSTM(data_width, opt.hs[l], opt.rho))
	rnn:add(nn.SoftSign())
	for i=1, opt.rl-1 do
		l = l+1
		rnn:add(nn.FastLSTM(opt.hs[l-1], opt.hs[l], opt.rho))
		rnn:add(nn.SoftSign())
	end

	model:add(nn.SplitTable(1, 2))
	model:add(nn.Sequencer(rnn))
	model:add(nn.SelectTable(-1))
	--Dense layers
	for i=1, opt.dl do
		l = l+1
		model:add(nn.Linear(opt.hs[l-1], opt.hs[l]))
		model:add(nn.SoftSign())
	end
	--Output layer
	model:add(nn.Linear(opt.hs[l], 1))
	model:add(nn.Sigmoid())--ReLU was a bad idea

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
criterion = nn.MSECriterion(true)
if opt.opencl then criterion:cl() end

data = create_dataset(opt.d, true, opt.ds)
data = normalize_col(data, 89)

totlen = get_total_len(data)

print(curr_ep)
print(start_ep)

train(optim.adagrad)

if opt.o ~= '' then
	save(model, opt.o)
end
