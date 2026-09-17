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
we model the discrepancy as the nominal model underestim

note from claude , we are underestimating the length in order to mimic the incomplete dynamics we might have in a specific example 

ating the true pendulum length, motivated by [brief physical reasoning]; we do not investigate the overestimation case, which may behave differently given the asymmetric role of length in the dynamics." That last clause is actually a nice, honest limitation to flag — it shows awareness that the choice matters, without requiring you to test both (which would double your Week 3 workload for a secondary question).
# training methodology and outcomes 
we train the neural networks and compare how well it predicts to find the empirical error e so we have a reference for the theoretical model as well 

# what we have built so far 

dynamics.py — true nonlinear pendulum physics, built as a reusable function (build_dynamics_functions) taking length as a parameter, discretized via RK4. Verified correct (open-loop fall simulation matched physical expectations).
nominal_mpc.py — a working do-mpc controller using the true dynamics: model, cost function, torque constraints, simulator, closed-loop test loop. Verified: smoothly stabilizes the pendulum from various initial conditions, no oscillation or divergence.
data_generation.py — generates training data via random-torque excitation across 18 varied initial conditions, producing 1800 (state, input, next-state) triples, saved to disk.
residual_dynamics.py — builds five "wrong" nominal models (5%, 15%, 30%, 50%, 75% length underestimation, using your existing dynamics function), computes the true residual (true next-state minus wrong-model prediction) for every data point at every discrepancy level, saves results.
residual_nn.py — defines a feedforward NN, trains one independently per discrepancy level (with proper 80/20 train/validation split), measures validation RMSE per level. Result: RMSE grows from 0.012 (5%) to 0.331 (75%), roughly monotonic, with a sharp jump at the highest level — your first genuine empirical finding.

#we have caluclated the terminal ingredients by solving the discrete algebraic ricatti equation and have snured stability for all 5 discrepancy levels

we will now need to implement a simplified terminal set in order to check the theoretical guarantees for stability of the project 

# linearization region 

we choose a conservative alpha to make sure we have a linear region of the system 

polytope W is calulcated using this conservative alpha( Linear region of system) 

# polytope W is computed empirically from the residuals 
estimated from the already computed residuals 
we generate a new data set to find the polytope W empirically 


"Physical parameters (mass, length, damping) and actuator limits were chosen as representative values for a small benchmark pendulum system, consistent with common practice in the learning-based MPC literature (e.g., Anderson's report, [GRU-MPC paper]), rather than calibrated to a specific real device. The qualitative relationships we study — how discrepancy severity affects residual-learning performance and stability — are not expected to depend sensitively on these particular choices."
