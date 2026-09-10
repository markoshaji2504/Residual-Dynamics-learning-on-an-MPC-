from Dynamics import build_dynamics_functions
import numpy as np

L_TRUE = 0.5
discrepancy_levels = [0.05, 0.15, 0.30, 0.50, 0.75]

nominal_models = {}
#create the nominal models for each discrepancy level described above 
for level in discrepancy_levels:
    l_wrong = L_TRUE * (1 - level)
    f_wrong, F_wrong = build_dynamics_functions(l=l_wrong)
    nominal_models[level] = F_wrong
#we end up with the nominal models array which contains all the dynamics for the base models that we will enchance witht the neural networks
#printing to make sure we get a logical output 
#print(nominal_models)
#print(nominal_models[0.30])

#load the data of the actual model for each discrepancy level in order to calculate the residual for each discrepancy level

data = np.load('data/pendulum_dataset.npz')
states = data['states']
inputs = data['inputs']
next_states_true = data['next_states']

residuals = {}

for level, F_wrong in nominal_models.items():
    predicted_wrong = np.array([
        np.array(F_wrong(states[i], inputs[i])).flatten()
        for i in range(len(states))
    ])
    residual = next_states_true - predicted_wrong
    residuals[level] = residual

print(residuals[0.30].shape)
print(residuals[0.30][:5])

#save the results to data 
np.savez('data/residuals.npz', 
          states=states, 
          inputs=inputs, 
          **{f'residual_{level}': residuals[level] for level in discrepancy_levels})