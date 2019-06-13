from retro_contest.local import make

# Mostly just moves to the right

def main():
    totalReturn = 0
    env = make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')
    obs = env.reset()
    while True:
        inx, iny, inc = env.observation_space.shape
        inx = int(inx/8)
        iny = int(iny/8)
        action = env.action_space.sample()
        action[7] = 1
        ob, reward, done, info = env.step(action)
        xpos = info['x']
        ypos = info['y']
        print('(%d, %s)' %(xpos,ypos))
        totalReturn += reward
        env.render()
        if done:
            print('episode complete, return =', totalReturn)
            env.reset()
            totalReturn = 0


if __name__ == '__main__':
    main()
