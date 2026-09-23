# Residual Learning for Robust MPC Under Model Discrepancy: An Empirical Study on an Inverted Pendulum

## Introduction

Model Predictive Control (MPC) relies on an accurate model of the system it is
controlling. In many real-world settings, obtaining such a model exactly is
difficult or impossible — the true dynamics may be too complex, or key
physical parameters (mass, length, friction) may be only approximately known.
One approach to this problem is *learning-based MPC*: using a neural network
to learn a correction to an imperfect nominal model, rather than relying on
the nominal model alone.

This project investigates a specific, practical question: **when a robust
MPC controller is built around a deliberately incorrect nominal model, does
adding a neural-network-learned correction (residual learning) meaningfully
improve closed-loop stability — and if so, over what range of model error?**

We study this question on a simple inverted pendulum, a canonical benchmark
system in control theory. We deliberately introduce error into the
controller's belief about the pendulum's length, at five severities (5%,
15%, 30%, 50%, and 75% underestimation), and compare two conditions at each
severity: a robust MPC using the incorrect nominal model alone, and the same
robust MPC augmented with a neural network trained to predict and correct
the resulting model error.

Rather than deriving a new theoretical stability guarantee, we adopt and
apply the stability framework of Learning-Based Model Predictive Control
(LBMPC) [Aswani et al.], in which safety is derived from a nominal model
with bounded uncertainty, decoupled from the accuracy of any learned
correction. We state clearly where we simplify this framework relative to
its full theoretical treatment, and rely on direct closed-loop simulation
as our primary evidence of practical stability.

Our central finding is that residual learning's benefit is **not uniform
across the range of model error we tested**: it offers little or no benefit
at low discrepancy (5-15%), where there is little error to correct; it
provides substantial, clear benefit at moderate-to-high discrepancy
(30-50%), successfully stabilizing cases where the uncorrected controller
fails outright; and at the most extreme level we tested (75%), it improves
but does not fully resolve the failure, with the corrected system settling
into a damped oscillation rather than either full recovery or complete
failure.
## Related Work

Learning-based MPC spans a range of approaches, well summarized by Hewing
et al. [survey], who categorize the field by what is learned (a full
dynamics model, a residual correction, or a control policy) and how
safety guarantees are preserved. This work is situated in the *residual
learning* branch of that taxonomy: rather than learning the full system
dynamics from scratch, we assume access to a nominal, but incorrect,
model, and learn only the correction needed to account for its error.

**Learning-Based MPC (LBMPC).** The theoretical framing adopted here
follows Aswani, Gonzalez, Sastry, and Tomlin's LBMPC framework, which
decouples safety from performance: a nominal model with a bounded
uncertainty set provides a deterministic stability guarantee, independent
of how — or how well — a second, learned "oracle" model corrects it. This
decoupling is what permits a *static*, offline-trained neural network
(rather than LBMPC's original online-adaptive oracle) to remain a valid
instantiation of the same framework.

**Closest prior implementation.** Anderson's report on LBMPC for an
inverted pendulum is the closest prior implementation to this study: it
applies the LBMPC framework to the same benchmark system and empirically
compares LQR, robust MPC, and LBMPC. This work extends that comparison in
two respects: it systematically varies the *severity* of nominal-model
error across five discrepancy levels rather than a single case, and it
pairs each level with a direct closed-loop comparison against the same
robust controller *without* the learned correction, isolating the
correction's specific contribution to stability.

**Black-box alternatives.** A separate line of work learns the full
system dynamics as a black box rather than a residual correction on a
nominal model, with accompanying stability guarantees (e.g. GRU-based and
deep-learning-based MPC formulations). Residual learning is adopted here
instead, as it requires substantially less training data to achieve
comparable accuracy and aligns directly with the LBMPC theoretical
framework this study builds on.

**Contribution.** This work provides a systematic, quantified
characterization of the range of nominal-model error over which residual
learning improves closed-loop stability under a fixed robust MPC
framework, via a direct paired comparison across five discrepancy
severities — a comparison not present in the prior work above.


## Method

### System

We study a simple, directly-actuated inverted pendulum: a point mass at
the end of a rigid arm, with a torque applied at the pivot. The
continuous-time dynamics are:

  θ̈ = (m g l sin(θ) + u − b θ̇) / (m l²)

where θ is the angle from upright (θ = 0, the unstable equilibrium), θ̇ is
angular velocity, u is the applied torque, and m, l, b, g are mass,
length, damping, and gravitational acceleration respectively. We use
m = 1.0 kg, l = 0.5 m, b = 0.1, g = 9.81 m/s², and a discrete-time step
of dt = 0.02 s obtained via 4th-order Runge-Kutta integration.

### Nominal MPC Baseline

We first implement a standard MPC controller using the exact (true)
system dynamics, to validate the modeling, cost function, and
constraint setup before introducing any model error or learning. This
controller successfully stabilizes the pendulum from a range of initial
conditions and serves as the source of training data described below.

### Inducing Model Discrepancy

To study the effect of model error, we construct a *nominal* model that
consistently underestimates the pendulum's true length by a fixed
percentage — 5%, 15%, 30%, 50%, and 75% — while the true simulated system
retains its correct length throughout. This choice isolates a single,
interpretable source of model error whose severity can be precisely
controlled, rather than introducing multiple simultaneous sources of
uncertainty.

### Data Generation

Training data is generated by simulating the true dynamics under
randomized torque excitation, sampled uniformly across a bounded range,
from a spread of initial conditions covering both near-equilibrium and
moderately displaced states. This yields a dataset of (state, input,
next-state) triples used both to train the residual network (below) and
to compute empirical residual error bounds.

### Residual Neural Network

For each discrepancy level, a feedforward neural network (two hidden
layers of 32 units, ReLU activations) is trained to predict the residual
— the difference between the true next state and the nominal model's
prediction — from the current state and input. Each network is trained
independently per discrepancy level, using an 80/20 train/validation
split, with validation RMSE reported as the network's empirical accuracy
at that level.

### Terminal Ingredients

To connect the empirical setup to the LBMPC theoretical framework, we
construct the terminal ingredients required for its stability argument:
the nominal model is linearized about the upright equilibrium (via
automatic differentiation), and a terminal cost matrix P and terminal
feedback gain K are obtained by solving the discrete algebraic Riccati
equation. A terminal set is defined as a conservatively small sublevel
set of the resulting quadratic cost, chosen without a rigorous derivation
of its maximal valid size (see Limitations).

### Uncertainty Bound and Simplified Feasibility Check

The LBMPC stability guarantee requires a bound, W, on the nominal model's
uncertainty. We estimate W empirically as the maximum observed residual,
per state dimension, over a dataset restricted to a bounded neighborhood
of the equilibrium — since the full training dataset's broader excitation
range included states outside the region relevant to this analysis. As a
practical, necessary (but not sufficient) proxy for the full invariant-set
feasibility analysis prescribed by the theory, we check whether the
terminal controller's corrective action for this worst-case disturbance
remains within the actuator's torque limit.

### Robust Model Predictive Controller

The controller is implemented as a multi-stage (scenario-based) robust
MPC, in which the nominal model's length parameter is treated as
uncertain and represented by three discrete scenarios (the nominal value,
and ± a fixed margin) at the first branching step of the horizon. The
controller's terminal cost uses the Riccati-derived matrix P for the
corresponding discrepancy level. In the NN-corrected condition, the
residual network's prediction — reconstructed symbolically to allow
integration with the optimization framework — is added to the nominal
model's prediction at every step of the controller's internal dynamics.

### Closed-Loop Evaluation

For each of the five discrepancy levels, we run two closed-loop
simulations of 18 seconds (900 steps): one using the robust MPC with the
discrepant nominal model alone, and one using the same controller with
the residual correction added. Both are evaluated against the same true
simulated dynamics, from the same initial condition (θ = 0.3 rad,
θ̇ = 0 rad/s). This yields ten trajectories in total, forming the basis
of the results presented below.