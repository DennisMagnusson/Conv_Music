require 'xlua'
if opt.opencl then
	require 'cltorch'
	require 'clnn'
else
	require 'torch'
	require 'nn'
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
end





