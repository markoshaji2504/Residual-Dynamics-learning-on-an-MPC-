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

### Plain MPC Baseline and Margin Sensitivity Check

To isolate the specific contribution of the multi-stage robustness
mechanism from that of the residual correction, we additionally evaluate
a plain (single-scenario, n_robust = 0) MPC using the same discrepant
nominal model, at each discrepancy level. We also verify that our
multi-stage implementation responds correctly to its uncertainty margin
by testing a substantially wider margin (±30%, versus ±5% used
throughout the main study) at one discrepancy level, confirming the
mechanism produces measurably different behavior when the margin is
large enough to matter.

### Closed-Loop Evaluation

For each of the five discrepancy levels, we run two closed-loop
simulations of 18 seconds (900 steps): one using the robust MPC with the
discrepant nominal model alone, and one using the same controller with
the residual correction added. Both are evaluated against the same true
simulated dynamics, from the same initial condition (θ = 0.3 rad,
θ̇ = 0 rad/s). This yields ten trajectories in total, forming the basis
of the results presented below.

## Results

### Residual Learnability

Figure 1 shows the trained neural network's validation RMSE at each
discrepancy level. Prediction error increases gradually through 50%
discrepancy (RMSE 0.012 to 0.077), then rises sharply at 75% (RMSE
0.339) — a roughly 4-5x jump compared to the increase seen at any prior
step. This suggests 75% discrepancy represents a qualitatively harder
regime for residual learning, not simply a continuation of the trend
observed at lower severities.

### Closed-Loop Stability

Figure 2 shows the closed-loop trajectory of the pendulum angle (θ) over
an 18-second window, for three conditions at each of five discrepancy
levels: a plain MPC using the discrepant nominal model, a robust
(multi-stage) MPC using the same model, and the robust MPC augmented
with the residual neural network correction.

Across all five levels, the plain and robust (nominal-only) conditions
are visually indistinguishable, indicating the multi-stage uncertainty
scenarios (±5% margin) contributed negligibly to closed-loop behavior at
this margin size; we return to this observation, and its verification,
in the Discussion.

The residual-corrected condition shows a distinct, non-monotonic pattern
across discrepancy severity:

- **At 5% and 15% discrepancy**, the NN-corrected trajectory converges
  more slowly than the uncorrected baseline, settling at a small but
  non-zero final angle (≈0.02-0.14 rad) where the uncorrected baseline
  converges essentially to zero.
- **At 30% discrepancy**, the NN-corrected trajectory converges past
  zero to a small negative angle (≈-0.07 rad), while the uncorrected
  baseline remains essentially static around 0.17 rad — a substantial,
  qualitative improvement.
- **At 50% discrepancy**, the uncorrected baseline diverges completely,
  settling near θ ≈ 2.28 rad; the NN-corrected controller converges
  cleanly to near zero — the clearest single demonstration of the
  correction's benefit.
- **At 75% discrepancy**, the uncorrected baseline settles at the
  downward equilibrium (θ ≈ π); the NN-corrected controller does not
  reach upright, but converges via a damped oscillation to an
  intermediate angle (≈1.5-1.75 rad) — a partial, but incomplete,
  improvement.

Figure 3 summarizes the final angular deviation at t = 18s across all
three conditions and discrepancy levels. Figure 4 shows the phase-space
trajectory (θ vs. θ̇) for the 75% discrepancy case specifically,
illustrating the qualitative difference between the uncorrected
controller's single-arc convergence to the downward equilibrium and the
NN-corrected controller's damped oscillatory approach to an intermediate
point.

### Simplified Feasibility Check

Applying the necessary-condition feasibility check described in Methods
(worst-case corrective torque vs. actuator limit), all five discrepancy
levels were found feasible, including 75%, where the closed-loop
simulation nonetheless failed to reach the upright target. This
discrepancy between the simplified check's prediction and the observed
closed-loop outcome is addressed in the Discussion.


## Discussion

### The Benefit of Residual Learning Is Not Uniform

The central finding of this study is that residual learning's practical
value depends strongly on the severity of the underlying model
discrepancy. At low discrepancy (5-15%), there is little systematic
error for the network to correct, and the learned correction appears to
introduce a small amount of prediction noise that mildly *slows*
convergence relative to the uncorrected baseline — a modest but
consistent negative result. At moderate-to-high discrepancy (30-50%),
the correction provides substantial, unambiguous benefit, most clearly
at 50%, where it is the difference between complete failure (divergence
to the wrong equilibrium) and clean convergence to the target. At the
most extreme level tested (75%), the correction meaningfully improves
the outcome — replacing a static failure at the downward equilibrium
with a damped oscillation toward an intermediate angle — but does not
fully resolve it.

This pattern is consistent with, and corroborated by, the residual
network's own validation error (Figure 1): the sharp increase in RMSE
between 50% and 75% discrepancy directly foreshadows the point at which
closed-loop performance also transitions from full recovery to partial
recovery. The consistency between these two independently measured
quantities — open-loop prediction accuracy and closed-loop control
outcome — strengthens confidence that both are capturing a genuine
property of the system, rather than measurement noise.

### The Robustness Margin Contributed Negligibly at the Tested Setting

The near-identical performance of the plain and robust (multi-stage)
MPC formulations, observed across all five discrepancy levels, might
suggest a flaw in the robust implementation. We verified this is not
the case: a supplementary test using a substantially wider uncertainty
margin (±30%, rather than the ±5% used throughout the main study)
produced a measurably different, and more conservative, closed-loop
outcome, confirming the multi-stage mechanism responds correctly to its
configured margin.

The negligible effect observed at the ±5% margin instead reflects the
relationship between that margin and the discrepancy severities under
study: because ±5% is small relative to discrepancies of up to 75%, the
three scenarios considered by the optimizer at each solve are close
enough to one another that little genuine disagreement exists for the
controller to hedge against. As a result, ordinary MPC feedback — which
both the plain and robust formulations share — accounts for the large
majority of the stabilization observed in the nominal-only baseline,
and the residual correction, not the robustness margin, is the primary
source of the improvements reported above. This finding is itself
useful: it isolates residual learning as the dominant contributing
factor in this study's results, independent of the specific robust MPC
formulation's design.

### The Simplified Feasibility Check Understates Real-World Failure

The necessary-condition feasibility check (Methods) found all five
discrepancy levels technically feasible under actuator constraints,
including 75% — a level at which the closed-loop simulation clearly
fails to reach the control objective. This gap illustrates a known
limitation of the simplified check: it evaluates only a single
worst-case correction at one instant, and does not account for
disturbances compounding over the full prediction horizon, nor for the
system's actual nonlinear behavior far from the equilibrium about which
the check's linear quantities (K, P) were derived. The closed-loop
simulation, not the simplified check, should be treated as the primary
evidence of practical stability in this study.

### Limitations

This study makes several deliberate simplifications relative to the
full LBMPC theoretical framework, made explicit here for transparency:

- The terminal set size (α) was chosen conservatively rather than
  rigorously derived from the nonlinear system's Taylor-remainder error
  bound.
- The uncertainty bound (W) was estimated empirically from a dataset
  restricted to a bounded neighborhood of the equilibrium, rather than
  derived analytically.
- The full disturbance-invariant set construction required for LBMPC's
  formal stability guarantee was not implemented; the necessary-condition
  feasibility check used here is a practical proxy, not a substitute for
  the complete guarantee.
- The residual network's uncertainty bound (used to justify the robust
  formulation) was held fixed across the NN-corrected and nominal-only
  conditions, rather than reduced to reflect the NN's improved accuracy;
  this may understate the potential benefit of combining residual
  learning with an appropriately-tuned robustness margin.
- Results are reported for a single initial condition (θ = 0.3 rad);
  generalization to other starting states was not tested.

### Future Work

Natural extensions include: deriving the terminal set size and
uncertainty bound analytically rather than conservatively; implementing
the full tube-MPC invariant-set construction to obtain a rigorous rather
than necessary-condition-only stability guarantee; extending the
residual learning approach to a second, more complex nonlinear system;
and exploring an online-adaptive variant of the residual correction, in
which the network continues to update using data collected during
closed-loop operation, as in the original LBMPC formulation.

## Conclusion

This study empirically characterized the practical value of residual
learning for a robust MPC controller operating under a range of
deliberately induced nominal-model errors. Rather than deriving a new
theoretical stability guarantee, we applied the LBMPC framework's
existing decoupling of safety and performance, implemented a simplified
necessary-condition feasibility check in place of its full invariant-set
construction, and relied on direct closed-loop simulation as our primary
evidence of practical stability — with each simplification stated
explicitly.

We found that residual learning's benefit is neither absent nor
universal: it provides no advantage, and a small disadvantage, at low
model discrepancy; substantial, unambiguous benefit at moderate-to-high
discrepancy, rescuing cases the uncorrected controller fails outright;
and partial, incomplete benefit at the most severe discrepancy tested.
This non-uniform pattern, corroborated independently by the network's
own prediction accuracy, is the central empirical contribution of this
work, and we hope it offers a useful, honestly-scoped data point for
understanding when learning-enhanced control is — and is not —
worthwhile.