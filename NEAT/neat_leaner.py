import os
import neat
import multiprocessing
import datetime
import gymnasium as gym
import numpy as np
import random
from tensorboardX import SummaryWriter
import time
runs_per_net = 35
log_dir = "tensorboard_logs/neat"  # Customize log directory
global_summary_writer = SummaryWriter(log_dir)

generation_count = 0

time_of_start = time.time()

def eval(genomes,config):
    global generation_count
    global biggest_player
    nets = []
    ge = []
    biggest_player = genomes[0][1]
    biggest_player.fitness = -200
    for genome_id, genome in genomes:

        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        ge.append(genome)

        # Runs the simulation and defines the fitness
    for i,n in enumerate(nets):
        final_reward = 0
        for run_nr in range(runs_per_net):
            env = gym.make("LunarLander-v2")
            observation, info = env.reset()
            done = False
            while not done:
                
                action = np.argmax(n.activate(observation)) # Last two are just "landed, booleans" - We will receive that info from rewards.
                observation, reward, terminated, truncated, info = env.step(action)
                final_reward = final_reward + reward
                if terminated or truncated:
                    done = True
            global_summary_writer.add_scalars("NEAT/Rewards", {f'net {i}':(final_reward/(run_nr+1))}, run_nr+(generation_count*runs_per_net))
            env.close()
        final_reward = final_reward / runs_per_net
        

        if biggest_player.fitness < final_reward:
            biggest_player = ge[i]
            biggest_player.fitness = final_reward
        ge[i].fitness = final_reward
    all_fitnesses = []
    for n in ge:
        all_fitnesses.append(n.fitness)
    global_summary_writer.add_scalar("NEAT/Avarage/Reward",np.mean(all_fitnesses),generation_count)
    global_summary_writer.add_scalar("NEAT/Avarage/Reward/time",np.mean(all_fitnesses),(time.time() - time_of_start))
    generation_count = generation_count+1

def visual_sim(genome,config,runs):
    n = neat.nn.FeedForwardNetwork.create(genome, config)
    for _ in range(runs): #Give you x tries.
        env = gym.make("LunarLander-v2",render_mode="human")
        observation, info = env.reset()
        final_reward = 0
        done = False
        while not done:
            action = np.argmax(n.activate(observation))
            observation, reward, terminated, truncated, info = env.step(action)
            final_reward = final_reward + reward
            if terminated or truncated:
                done = True
        env.close()
        print(f"This run gave a reward of: {final_reward}")

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir,"config.txt")
    config = neat.config.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation,config_path)


    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    winner = p.run(eval,170)
    print(f"Winner: {winner}")
    print(f"It took {(time.time()-time_of_start)/60} minutes to compleet!")
    print("Running 10 visual loops to show you what has been done!")
    visual_runs = int(input("How many runs would you like to see?\n"))
    visual_sim(winner,config,visual_runs)
    with open("output.txt", "w") as txt_file:
        txt_file.write(str(winner)) 

    print(f"Saved biggest player has fitness level of: {biggest_player.fitness}")
    visual_runs = int(input("How many runs would you like to see with the BP?\n"))
    visual_sim(biggest_player,config,visual_runs)