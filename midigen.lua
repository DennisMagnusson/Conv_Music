local midi = require 'MIDI'

function generate(pattern, ...)
	local arg = table.pack(...)
	local score = {1000, {}}
	
	local time = 0

	--pattern = to_abs_time(pattern)
	for k, frame in pairs(pattern) do
		local tones = {}
		for n=1, 88 do
			if frame[n] ~= 0 then tones[#tones+1] = n end
		end

		for i, tone in pairs(tones) do
			score[2][#score[2]+1] = {'note', time, 500, 1, tone+20, 53}--Duration=500, v=53
		end
		time = time + 500
	end
	
	table.insert(score[2], 1, {'control_change', 0, 2, 7, 100})
	table.insert(score[2], 1, {'control_change', 0, 2, 10, 64})
	table.insert(score[2], 1, {'control_change', 0, 1, 7, 100})
	table.insert(score[2], 1, {'control_change', 0, 1, 10, 64})
	table.insert(score[2], 1, {'control_change', 0, 1, 91, 127})
	table.insert(score[2], 1, {'set_tempo', 0, 1000000})

	if arg[1] then
		local file = assert(io.open("generated/" .. arg[1], 'w'))
		file:write(midi.score2midi(score))
		file:close()
		return
	else return score end
end

--Unused, for now
function to_abs_time(r)
	local t = 0
	for k, frame in pairs(r) do
		t = t + frame[92]
		frame[92] = t
	end
	return r
end
