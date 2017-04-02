require 'xlua'

function get_args(arg, conv)
	arg = arg or {}
	conv = conv or true

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

	return d
end
