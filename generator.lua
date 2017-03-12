require 'midigen'
require 'midiparse'
require 'torch'
require 'nn'
require 'rnn'
json = require 'json'
require 'cltorch'

cmd = torch.CmdLine()
cmd:option('-o', '', 'Output file name')
cmd:option('-model', '', 'Model file name')
cmd:option('-temperature', 0, 'Temperature')
cmd:option('-firstnote', 41, 'First note index 1-88')
cmd:option('-len', 100, 'Length of the notes')
cmd:option('-k', 0, 'k-value')
cmd:option('-start', '', 'File to use the first rho timesteps on')
cmd:option('-print', false, 'Print the result')
opt = cmd:parse(arg or {})

local prev_notes = {}

function denormalize_col(r, col)
	local min = meta[col..'min']
	local max = meta[col..'max']
	for i=1, #r do
		r[i][col] = math.floor(math.exp(r[i][col]*(max-min)+min)-1)
	end
	return r
end

function create_song()
	math.randomseed(os.time())

	local song = torch.Tensor(opt.len, data_width)
	local x = torch.Tensor()
	if opt.start == '' then
		x = torch.zeros(meta['rho'], data_width)
		x[meta['rho']][opt.firstnote] = 1
		x[meta['rho']][93] = 0.2
		x[meta['rho']][91] = 1
	else
		x = torch.Tensor(load_start(opt.start))
	end
	for i=1, opt.len do
		local pd = model:forward(x)--Probability distribution... thing
		pd = pd:reshape(data_width)
		local frame = sample(pd)
		--Push everything back one step
		for u=2, meta['rho'] do
			x[u-1] = x[u]
		end
		--Add frame
		x[meta['rho']] = frame
		--Save frame
		song[i] = torch.Tensor(frame)
	end
	local r = torch.totable(song)
	r = denormalize_col(r, 92)
	r = denormalize_col(r, 93)

	if opt.o ~= '' then
		generate(r, opt.o)
	end
	if opt.print then
		print(get_notes(song))
	end
end

function sample(frame)
	local r = torch.zeros(88)
	for i=1, 88 do
		r[i] = frame[i]
	end
	
	--Prevent the same note from being played over and over
	local a = prev_notes[#prev_notes]
	if #prev_notes > 5 then
		for i=#prev_notes, #prev_notes-4, -1 do
			if prev_notes[i] ~= a then goto smple end
			a = prev_notes[i]
		end
		r[prev_notes[#prev_notes]] = 0
	end
	::smple::

	local sum = opt.k
	if sum == 0 then sum = torch.sum(r) end

	local tmp = opt.temperature
	if tmp == 0 then tmp = 1 end
	
	r = torch.exp(torch.log(r) / tmp)
	r = r / torch.sum(torch.exp(r))
	r = r*(sum / torch.sum(r))

	local max = torch.max(r)
	
	local empty = true
	for _, i in pairs(torch.totable(torch.randperm(88))) do --Random order
		local rand = math.random()
		if r[i] > rand and (r[i] > 0.1 or r[i] == max) then --Ignore low probabilities
			r[i] = 1
			prev_notes[#prev_notes+1] = i
			--Prevent awful doubles
			r[i-1] = 0
			r[i-2] = 0
			r[i+1] = 0
			r[i+2] = 0
			empty = false
		else
			r[i] = 0
		end
	end
	if empty then return sample(frame) end

	for i=1, 88 do
		frame[i] = r[i]
	end
	--Pedal
	if frame[89] > 0.7 then
		frame[89] = 1
		frame[90] = 0
	elseif frame[90] > 0.7 then
		frame[89] = 0
		frame[90] = 1
	else
		frame[89] = 0
		frame[90] = 0
	end
	return frame
end

function normalize(r, col)
	r[col] = ((math.log(r[col]+1)-meta[col..'min'])/(meta[col..'max']-meta[col..'min']))
	return r
end

function load_start(filename)
	local start = parse(filename)
	local r = {}
	for i=1, meta['rho'] do
		r[i] = normalize(normalize(start[i], 92), 93)
	end
	return r
end

function get_notes(r)
	local notes = {}
	for i=1, opt.len  do
		notes[i] = {}
		for u=1, data_width do
			if r[i][u] ~= 0 then
				notes[i][#notes[i]+1] = u
			end
		end
	end
	return notes
end

model = torch.load(opt.model):double()
data_width = 93
file = assert(io.open(opt.model..".meta", 'r'))
meta = json.decode(file:read('*all'))
file:close()
create_song()
