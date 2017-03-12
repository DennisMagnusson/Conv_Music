local midi = require 'MIDI'

--[[
1-88	notes
89		pedal down
90		pedal up
91		velocity
92		delta time
93		duration
]]

function parse(filename)
	local file = assert(io.open(filename, 'r'))
	local m = midi.midi2ms_score(file:read("*all"))
	file:close()
	file = nil
	--Remove all non-notes or non-control_change events
	local notes = {}

	for k, event in pairs(m[2]) do
		if event[1] == 'note' or event[1] == 'control_change' then 
			notes[#notes+1] = event
		end
	end
	if #notes == 0 and m[3] then
		for k, event in pairs(m[3]) do
			if event[1] == 'note' or event[1] == 'control_change' then 
				notes[#notes+1] = event
			end
		end
	end


	notes = sort_by_time(notes)

	local r = {}
	for k, event in pairs(notes) do
		--Blend together if dt < 30 ms
		if #r > 1 and event[2]-notes[k-1][2] <= 30 then
			--Time
			r[#r][92] = r[#r][92] + event[2]-notes[k-1][2]
			--Pedal
			if event[1] == 'control_change' then
				if event[5] == 127 then r[#r][89] = 1
				elseif event[5] == 0 then r[#r][90] = 1 end
				goto EOL
			end
			
			--tone
			r[#r][event[5]-20] = 1
			--velocity
			if event[6]/127 > r[#r][91] then r[#r][91] = event[6]/127 end
			--duration
			if event[3] > r[#r][93] then r[#r][93] = event[3] end
			goto EOL
		end

		--Fill frame with zeros
		local frame = {}
		for i = 1, 93 do frame[i] = 0 end

		if event[1] == 'control_change' then
			if event[5] == 127 then frame[89] = 1
			elseif event[5] == 0 then frame[90] = 1 end
			--delta start_time
			if #r <= 1 then
				frame[92] = 0
			else 
				frame[92] = event[2] - notes[k-1][2]
			end
			r[#r+1] = frame
			goto EOL
		end
		
		--tone
		frame[event[5]-20] = 1
		--velocity
		frame[91] = event[6]/127
		--delta start_time
		if #r <= 1 then 
			frame[92] = 0 --First one is zero
		else
			frame[92] = event[2] - notes[k-1][2]
		end
		--duration
		frame[93] = event[3]
		if frame[93] < 1 then frame[93] = 1 end
		--Insert into r
		r[#r+1] = frame
		::EOL::
	end
	return r
end

function sort_by_time(r)
	local s = {}
	s[1] = r[1]
	for i=2, #r do
		local l = #s
		for u=1, #s do
			if r[i][2] < s[u][2] then 
				table.insert(s, u, r[i])
				goto cont
			end
		end
		if s[l+1] == nil then s[l+1] = r[i] end
		::cont::
	end
	return s
end
