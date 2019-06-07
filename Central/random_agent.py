import retro

def main():
    env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')
    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            obs = env.reset()


if __name__ == '__main__':
    main()