import casadi as ca 
#declare the variables for the pendulum parameters
m = 1.0    # mass, kg
l = 0.5    # length, m
b = 0.1    # damping coefficient
g = 9.81   # gravity, m/s^2
#initilize the symbolic variables 
theta=ca.SX.sym("theta")
theta_dot=ca.SX.sym("theta_dot")
u=ca.SX.sym("u")
x = ca.vertcat(theta, theta_dot)
#set up the system dynamics 
#angular acceleration
theta_ddot = (m * g * l * ca.sin(theta) + u - b * theta_dot) / (m * l**2)
#rate of change of vector 
x_dot = ca.vertcat(theta_dot, theta_ddot)
#creating the function to call to claculate vector x-dot 
f = ca.Function('f', [x, u], [x_dot])

#RK4 discritization of the system dynamics 

#discretize using RK4 integration
dt = 0.02  # timestep in seconds

k1 = f(x, u)
k2 = f(x + dt/2 * k1, u)
k3 = f(x + dt/2 * k2, u)
k4 = f(x + dt * k3, u)
x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

F = ca.Function('F', [x, u], [x_next])

# simulate a few steps forward with no control input, starting near upright TEST

#x_current = [0.3, 0.0]
#for step in range(10):
#    x_current = F(x_current, 0.0)
#    x_current = [float(x_current[0]), float(x_current[1])]  # convert back to plain numbers
#    print(f"step {step}: theta = {x_current[0]:.4f}, theta_dot = {x_current[1]:.4f}")



