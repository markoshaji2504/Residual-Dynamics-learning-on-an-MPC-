#create the data to train the Neural Networks 
from Dynamics import F
from nominal_mpc import u_max
import numpy as np

 #needs to bet he same as defined in the nominal mpc file 
def generate_random_input(n_steps, u_max):
    return np.random.uniform(-u_max, u_max, n_steps)
#generate random torque for excitation 
 

 #simulate the trajectory of the pendulum 
def simulate_trajectory(x0, n_steps, u_max):
    x = np.array(x0)
    inputs = generate_random_input(n_steps, u_max) #generate random inputs withing the max torque constraint 

    data = []
    for k in range(n_steps):
        u = inputs[k]
        x_next = np.array(F(x, u)).flatten() #call F from Dynamics.py to simulate the pendulum 
        data.append((x.copy(), u, x_next.copy()))
        x = x_next

    return data


#test for the simulate trajectory function 
#test_data = simulate_trajectory(x0=[0.3, 0.0], n_steps=10, u_max=5)
#for entry in test_data:
#    print(entry)

#creating the function that generates all of the training data for the neural networks 
def generate_dataset(initial_conditions, n_steps, u_max):
    all_data = []
    for x0 in initial_conditions:
        trajectory_data = simulate_trajectory(x0, n_steps, u_max)
        all_data.extend(trajectory_data)
    return all_data


initial_conditions = [
    [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0], [0.5, 0.0],
    [-0.1, 0.0], [-0.2, 0.0], [-0.3, 0.0], [-0.4, 0.0], [-0.5, 0.0],
    [0.2, 0.3], [0.2, -0.3], [-0.2, 0.3], [-0.2, -0.3],
    [0.4, 0.5], [0.4, -0.5], [-0.4, 0.5], [-0.4, -0.5],
]

n_steps = 100
u_max = 5
#testing the generate data set function 
dataset = generate_dataset(initial_conditions, n_steps, u_max)
#print(f"Total number of data points: {len(dataset)}")
#print("First entry:", dataset[0])
#print("Last entry:", dataset[-1])


#save the data generation file so we have the same data each time 
import numpy as np

# convert to arrays for easier saving/loading
states = np.array([entry[0] for entry in dataset])
inputs = np.array([entry[1] for entry in dataset])
next_states = np.array([entry[2] for entry in dataset])

np.savez('data/pendulum_dataset.npz', states=states, inputs=inputs, next_states=next_states)
loaded = np.load('data/pendulum_dataset.npz')
print(loaded['states'].shape)
print(loaded['inputs'].shape)
print(loaded['next_states'].shape)