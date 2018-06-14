from ddpg import DDPG
import gym
import matplotlib.pyplot as plt

avgn = 30

env = gym.make('MountainCarContinuous-v0')
agent = DDPG(env)

def get_steps(name):
    ckdir = agent.checkpoint_dir
    ckpoints = [480]
    maxep = max(ckpoints)
  
    for ckpoint in ckpoints:
        ckpointname = "Github_"+name+"_500/DDPG-PyTorch"+"ep{}.pth.tar".format(ckpoint)
        print(ckpointname)
        agent.loadCheckpoint(ckpointname)
        #agent.play(showdata=False)
        #print(agent.stepgraph)
        if ckpoint == maxep:
            n = agent.start -1 - avgn+1
            eps = list(range(n))
            avgs = [sum(agent.stepgraph[i:i+avgn])/avgn for i in eps]
            return eps, avgs

def play(name):

    ckpoints = list(range(0, 120, 20))

    for ckpoint in ckpoints:
        ckpointname = "Github_"+name+"_500/DDPG-PyTorch"+"ep{}.pth.tar".format(ckpoint)
        print(ckpointname)
        agent.loadCheckpoint(ckpointname)
        print("episode:", ckpoint)
        agent.play(showdata=False)
  
def show_one_model(name):
    eps, avgs = get_steps(name)
    plt.title('DDPG with '+name+' reward')
    plt.ylabel('number of steps to succeed')
    plt.xlabel('episode')
    plt.plot(eps, avgs, 'r', eps, )
    plt.savefig("DDPG_"+name+"_{}".format(avgn))
    plt.show()

def compare(names):
    plt.title('DDPG: comparison of different rewards')
    plt.ylabel('number of steps to succeed')
    plt.xlabel('episode')
    steps = [get_steps(name) for name in names]
    for i in range(len(names)):
        plt.plot(steps[i][0], steps[i][1], label=names[i])
    plt.legend()
    plt.savefig("compare_"+"_".join(names)+"_{}".format(avgn))
    plt.show()


#compare(['E', 'raw'])
play('raw')

env.close()
