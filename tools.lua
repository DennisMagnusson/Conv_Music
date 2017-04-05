require 'xlua'

function get_args(arg, conv)
	arg = arg or {}

	local cmd = torch.CmdLine()
	cmd:option('-d', '', 'Dataset directory')
	cmd:option('-vd', '', 'Validatation data directory')
	cmd:option('-ds', 0, 'Size of dataset (for benchmarking)')
	cmd:option('-o', '', 'Output filename')
	cmd:option('-ep', 1, 'Number of epochs')
	cmd:option('-bs', 128, 'Batch size')
	cmd:option('-drop', 0.0, 'Dropout probability')
	cmd:option('-lr', 0.0001, 'Learning rate')
	cmd:option('-lrd', 0.0, 'Learning rate decay')
	cmd:option('-wd', 0.0, 'Weight decay')
	cmd:option('-cpu', false, 'Use CPU')
	cmd:option('-dl', 1, 'Number of dense layers')
	cmd:option('-rho', 88, 'Sequence length')
	cmd:option('-hs', '64,64', 'Size of hidden layers')
	if conv then
		cmd:option('-ks', '13x1,13x5', 'Kernel sizes')
		cmd:option('-c', '2,5', 'Channels in layers')
	else
		cmd:option('-rl', 1, 'Number of recurrent layers')
	end

	local opt = cmd:parse(arg or {})
	opt.opencl = not opt.cpu

	opt.hs = parse_str(opt.hs)
	if conv then
		opt.c = parse_str(opt.c)
		opt.ks= parse_kernel_sizes(opt.ks)
	else
		if #opt.hs ~= opt.rl + opt.dl then
			assert(false, "Number of hiddensizes is not equal to number of layers")
		end
	end

	return opt
end

--Parse any string with multiple values
function parse_str(h)
	local ch = {}
	while true do
		if h:len() == 0 then break end
		local c = h:find(',') or h:len()+1
		local str = h:sub(1, c-1)
		h = h:sub(c+1, h:len())
		ch[#ch+1] = tonumber(str)
	end

	return ch
end

--Parse string with kernel sizes
function parse_kernel_sizes(str)
	local ks = {}
	while true do
		if str:len() == 0 then break end
		local c = str:find(',') or str:len()+1
		local x_index = str:find('x')
		local n1 = str:sub(1, x_index-1)
		local n2 = str:sub(x_index+1, c-1)--Right before the next comma
		ks[#ks+1] = {tonumber(n1), tonumber(n2)}
		str = str:sub(c+1, str:len())
	end

	return ks
end

function get_total_len(data)
	local c = 0
	for _, s in pairs(data) do
		c = c + s:size()[1]
	end
	return c
end

function create_dataset(dir, time, datasize)
	local d = {}

	for filename in lfs.dir(dir.."/") do
		if filename[1] == '.' then goto cont end
		if datasize ~= 0 and #d >= datasize then return d end
		local song = parse(dir..'/'..filename, time)
		if #song > 2 then
			d[#d+1] = torch.Tensor(song)
		end
		::cont::
	end
	--Trim if datasize is entered
	if opt.ds ~= 0 then
		local l = #d
		for i=opt.ds, l do
			d[i] = nil
		end
	end
	return d
end

--function train(model, optimizer, config, feval, params)
function train(optimizer)
	model:training()--Training mode
	math.randomseed(os.time())

	local optim_cfg = {learningRate=opt.lr, learningRateDecay=opt.lrd, weightDecay=opt.wd}
	local progress = -1

	for e=1, opt.ep do
		while curr_ep == start_ep+e do
			if progress ~= math.floor(100*(start_index/totlen)) then
				progress = math.floor(100*(start_index/totlen))
				xlua.progress(100*(e-1)+progress, 100*opt.ep)
			end

			optimizer(feval, params, optim_cfg)
			collectgarbage()
		end
	end

	model:evaluate() --Exit training mode
end

function reload_model(filename)
	local model = torch.load(opt.o)
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
	opt.ds = 0

	print(opt)

	curr_ep = meta['ep']+1
	start_ep = meta['ep']
	opt.lr = meta['lr']/(1+meta['lrdecay']*meta['ep'])--Restore decayed lr
	meta['ep'] = meta['ep'] + opt.ep
	print("opt.ep:", opt.ep, "meta.ep", meta.ep)
	logger = optim.Logger(opt.o..".log2")

	return model
end


function new_epoch()
	start_index = 1

	local prev_loss = loss
	loss = totloss/batches
	local delta = loss-prev_loss

	model:evaluate()
	validation_err = validate(model, opt.rho, opt.bs, opt.vd, criterion)
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

--TODO Needs testing
function create_batch(data, index, time)
	local song = torch.Tensor()
	local i = index
	local songindex = -1
	--Select song
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
	if time then output_width = 1
	else output_width = 88 end

	local x = torch.Tensor(opt.bs, opt.rho, data_width)
	local y = torch.Tensor(opt.bs, output_width)

	for u=1, opt.bs do
		::s::
		--Go to next song if finished
		if song:size()[1] < i+u+opt.rho+1 then
			--Break if data is end
			if songindex == #data then break end
			song = data[songindex+1]
			i=1
			goto s
		end
		
		for o = opt.rho, 1, -1 do
			x[u][o] = song[i+o+u]
		end
		if time then
			y[u] = song[i+u+opt.rho][89]
			x[u][opt.rho][89] = 0
		else
			y[u] = song[i+u+opt.rho+1]
		end

		--Transformations for conv
		if not time then
			x = x:transpose(2, 3)
			x = torch.reshape(x, opt.bs, 1, data_width, opt.rho)
		end

		if opt.opencl then
			x = x:cl()
			y = y:cl()
		end
	
	return {x, y}
end

function save(model, out)
	torch.save(out, model)
	local file = assert(io.open(out..".meta", 'w'))
	file:write(json.encode(meta))
	file:close()
	--Merge logs if needed
	if resume then 
		os.execute("cat "..out.."log2 >> "..out..".log")
	end
end
