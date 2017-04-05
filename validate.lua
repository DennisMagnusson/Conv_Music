require 'midiparse'
require 'lfs'
require 'rnn'
require 'optim'

if opt.opencl then
	require 'cltorch'
	require 'clnn'
else
	require 'torch'
	require 'nn'
end


function validate(model, rho, batchsize, dir, criterion, time)
	local valid_data = create_data(dir, time) --Faster than saving

	local width = -1
	if time then
		width = 89
	else
		width = 88
	end

	local toterr = 0
	local c = 0
	local bs = 1

	local i = 1

	while true do
		local batch = create_batch(valid_data, i, time)
		if batch == -1 then break end
		local x = batch[1]
		local y = batch[2]
		if opt.opencl then
			x = x:cl()
			y = y:cl()
		end

		local pred = model:forward(x)
		local err = torch.mean(torch.abs(y - pred))
		if err ~= err then--If err is nan
			break--Temporary solution, lol
		end

		toterr = toterr + err
		c = c + 1
		i = i + batchsize
	end

	collectgarbage()

	return toterr / c
end

function normalize(r, col)
	for i=1, #r do
		for u=1, r[i]:size()[1] do
			if r[i][u][col] > 4000 then r[i][u][col] = 4000 end
			r[i][u][col] = math.log(r[i][u][col]+1) / 8.294
		end
	end
	return r
end

function create_data(dir, time)
	local songs = {}
	for filename in lfs.dir(dir.."/.") do
		if filename:sub(1,1) == '.' then goto cont end
		local song = parse(dir.."/"..filename, time)
		songs[#songs+1] = torch.Tensor(song)
		::cont::
	end
	if time then
		songs = normalize(songs, 89)
	end
	return songs
end
