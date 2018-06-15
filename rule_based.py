import gym

env = gym.make('MountainCar-v0')

for episode in range(50) :
    env.reset()
    act = 0
    env.render()
    
    while True:
        s, reward, done, info = env.step(act)
        env.render()
        if done : break
        if s[1] < 0 : act = 0
        elif s[1] > 0 : act = 2
        else : act = 1
