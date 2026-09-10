stability theorem guaranty from LBMPC paper 


# Notes


we will compare raw discrepenacy vs for the NN correct for vs the theorems guaranteed bound D*
one expected insight is that even though for specific implementatins nominal mpc stablity might not be satisfied due to the error bound being exceeded the NN might be able to push the system into stability and convergence 

# setting parameters and tuning of MPC model 
cost function 

# creating the baseline model for the mpc and testing convergence
i wanted to create a baseline model to have a good cost function constraints etc 

# generating the training data by random torque excitations 
keep in mind that you might need to tweek the data generation functions
as of 08/09 we have 18 different trajectories with 100 steps each , and no limits on where the pendulum ends up 
so we have all the positions covered. It might be revealed that it would beneficial to have a constaint so we end up with more data along the range of convergence also max torque might need to be changed for this to happen 

# next step is training the neural networks using each discrepancy level 
i first built the wrong nominal model
and calculate the positions for each time step
compare with true nominal model to find the residual for each discrepancy level 
train the residual NN for each discrepancy level 


