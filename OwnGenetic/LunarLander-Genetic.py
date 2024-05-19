import gymnasium as gym
import numpy as np
import random 
import threading


class NeuralNetwork:
    fitness = 0
    previous_fitness = -300
    id = 0
    previous_weights = []
    def __init__(self,id, input_nodes, hidden_layer_nodes, output_node=4):
        self.id = id
        self.input_nodes = input_nodes
        self.hidden_layer_nodes = hidden_layer_nodes
        self.output_node = output_node
        self.activation_func = self._sigmoid

        # Initialize weights based on the network architecture
        self.weights = []
        if len(hidden_layer_nodes) == 0:  # Direct input-to-output connection
            self.weights.append(np.random.rand(input_nodes, output_node))
        else:
        # Weights between input and first hidden layer
            self.weights.append(np.random.rand(input_nodes, hidden_layer_nodes[0]))

        # Weights between hidden layers (if any)
        for i in range(len(hidden_layer_nodes) - 1):
            self.weights.append(np.random.rand(hidden_layer_nodes[i], hidden_layer_nodes[i + 1]))

        # Weights between last hidden layer and output
        self.weights.append(np.random.rand(hidden_layer_nodes[-1], output_node))
        self.previous_weights = self.weights
    def predict(self, inputs):

        inputs = np.atleast_2d(inputs)  # Ensure input is a 2D array
        activations = inputs  # Store activations for each layer

        # Forward propagation through hidden layers and output layer
        for i, w in enumerate(self.weights):
            activations = self.activation_func(np.dot(activations, w))
        return np.argmax(activations)  # Output of the last layer
  

    def _sigmoid(self,x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self,x):
        """Derivative of the sigmoid activation function."""
        return self._sigmoid(x) * (1 - self._sigmoid(x))

class Genetic_Algorithm():
    agents = []
    INPUT_NODES = 8
    HIDDEN_LAYERS = [6]
    OUTPUT_LAYER = 4

    top_fitness = 0
    top_weights = []
    def __init__(self,population_size,mutation_rate=0.1,illiterations=10):
        self.mutation_rate = mutation_rate
        self.illiterations = illiterations
        self.pop_size = population_size
        for i in range(population_size):
            self.agents.append(NeuralNetwork(i,self.INPUT_NODES,self.HIDDEN_LAYERS))
        
    
    def run_training(self):
        for i in range(self.illiterations):
            threads = []
            for agent in self.agents:
                # Create a thread for each agent for a simulation to run. passing along the agent itself.
                thread = threading.Thread(target=lambda arg: self._simulation(arg), args=(agent,))
                thread.start()
                threads.append(thread)

            # Wait for all threads to finish
            for thread in threads:
                thread.join()
            sel = self.selection()
            self.top_fitness = sel[0][1]
            self.top_weights = self.agents[sel[0][0]].weights

            self.create_ofspring(sel)
            print(f"Illiteration {i} completed. Highest so far: {sel[0]}")
            


            with open("Epoch", "a") as file_object:
                # Convert the number to string and add a newline character
                file_object.write(str(sel[0][1]) + "\n")

    def selection(self):
        # Check if mutation/child was better or to go back to an old parrent:
        
        # Find the top 20%
        score = []
        for a in self.agents:
            score.append((a.id,a.fitness))
        selection = sorted(score, key=lambda x: x[1],reverse=True)
        selection = selection[0:int(0.2*self.pop_size)]
        return selection

    def create_ofspring(self, selection):
        new_weights = []
        for i in range(self.pop_size):
            parent1_id = random.choice(selection)
            parent1 = self.agents[parent1_id[0]]
            parent2 = random.choice(self.agents)
            temperary_weight = parent1.weights
            i = random.randint(0,len(temperary_weight)-1)
            total_amount_of_weights = len(temperary_weight)*len(temperary_weight[0])*len(temperary_weight[0][0])
            for i in range(random.randint(0,total_amount_of_weights-1)): #Random amount of weights to exchange.
                i1 = random.randint(0,len(temperary_weight)-1)
                i2= random.randint(0,len(temperary_weight[i1])-1)
                i3 = random.randint(0,len(temperary_weight[i1][i2])-1) 
                if random.random() < self.mutation_rate:
                    temperary_weight[i1][i2][i3] = random.random()
                else:
                    temperary_weight[i1][i2][i3] = parent2.weights[i1][i2][i3]
            new_weights.append(temperary_weight)
        for a in self.agents:
            a.weights = new_weights[a.id]

            
    def _simulation(self,nn_instance,render=None):
        # Runs the simulation and defines the fitness
        env = gym.make("LunarLander-v2",render_mode=render)
        observation, info = env.reset()
        final_reward = 0
        for _ in range(1000):
            action = nn_instance.predict(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            print(reward)
            if terminated or truncated:
                observation, info = env.reset()
            final_reward = final_reward+reward
        nn_instance.fitness = final_reward
        env.close()

    def visual_simulation(self):
        # Shows the best canididate visually land.
        score = []
        for a in self.agents:
            score.append((a.id,a.fitness))
        selection = sorted(score, key=lambda x: x[1],reverse=True)
        self._simulation(self.agents[selection[0][0]],"human")

    def save_best_weights(self):
            # TODO: Make this work.
        score = []
        for a in self.agents:
            score.append((a.id,a.fitness))
        selection = sorted(score, key=lambda x: x[1],reverse=True)
        with open("output.txt", "w") as txt_file:
                txt_file.write(str(self.agents[selection[0][0]].weights)) 

if __name__ == "__main__":
    GA = Genetic_Algorithm(1,0.05,100)
    GA.run_training()
    #GA.visual_simulation()
    GA.save_best_weights()