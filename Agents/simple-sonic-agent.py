from retro_contest.local import make

# Mostly just moves to the right

def main():
    totalReturn = 0
    env = make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')
    obs = env.reset()
    while True:
        action = env.action_space.sample()
        action[7] = 1
        ob, reward, done, _ = env.step(action)
        totalReturn += reward
        env.render()
        if done:
            print('episode complete, return =', totalReturn)
            env.reset()
            totalReturn = 0


if __name__ == '__main__':
    main()
