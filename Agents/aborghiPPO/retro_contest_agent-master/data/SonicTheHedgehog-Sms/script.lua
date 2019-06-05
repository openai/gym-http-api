level_max_x = {
    ["level=0"] = 6400,
    ["level=1"] = 2600,
    ["level=3"] = 5300,
    ["level=4"] = 3800,
    ["level=6"] = 7050,
    ["level=9"] = 1700,
    ["level=10"] = 1900,
    ["level=11"] = 700,
    ["level=12"] = 4600,
    ["level=13"] = 3900,
    ["level=15"] = 3900,
    ["level=16"] = 1600,
}

function clip(v, min, max)
    if v < min then
        return min
    elseif v > max then
        return max
    else
        return v
    end
end

prev_lives = 3

function contest_done()
    if data.lives < prev_lives then
        return true
    end
    prev_lives = data.lives

    if calc_progress(data) >= 1 then
        return true
    end

    return false
end

offset_x = nil
end_x = nil

function calc_progress(data)
    if offset_x == nil then
        offset_x = -data.x
        local key = string.format("level=%d", data.level)
        end_x = level_max_x[key] - data.x
    end

    local cur_x = clip(data.x + offset_x, 0, end_x)
    return cur_x / end_x
end

prev_progress = 0
frame_count = 0
frame_limit = 18000

function contest_reward()
    frame_count = frame_count + 1
    local progress = calc_progress(data)
    local reward = (progress - prev_progress) * 9000
    prev_progress = progress

    -- bonus for beating level
    if progress >= 1 then
        reward = reward + (1 - clip(frame_count/frame_limit, 0, 1)) * 1000
    end
    return reward
end
