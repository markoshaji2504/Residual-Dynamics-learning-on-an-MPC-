import casadi as ca 
#declare the variables for the pendulum parameters
#relevant parameters are set by the function we define 
#new way of defining dynamics in a function we can call 

def build_dynamics_functions(l, m=1.0, b=0.1, g=9.81, dt=0.02):
    theta = ca.SX.sym("theta")
    theta_dot = ca.SX.sym("theta_dot")
    u = ca.SX.sym("u")
    x = ca.vertcat(theta, theta_dot)

    theta_ddot = (m * g * l * ca.sin(theta) + u - b * theta_dot) / (m * l**2)
    x_dot = ca.vertcat(theta_dot, theta_ddot)

    f = ca.Function('f', [x, u], [x_dot])

    k1 = f(x, u) #rk4 disciretization happening inside the function now 
    k2 = f(x + dt/2 * k1, u)
    k3 = f(x + dt/2 * k2, u)
    k4 = f(x + dt * k3, u)
    x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    F = ca.Function('F', [x, u], [x_next])

    return f, F


L_TRUE = 0.5
f, F = build_dynamics_functions(l=L_TRUE)
#RK4 discritization of the system dynamics 



# simulate a few steps forward with no control input, starting near upright TEST

x_current = [0.3, 0.0]
for step in range(10):
    x_current = F(x_current, 0.0)
    x_current = [float(x_current[0]), float(x_current[1])]  # convert back to plain numbers
    print(f"step {step}: theta = {x_current[0]:.4f}, theta_dot = {x_current[1]:.4f}")

#test is good works as of 09/09/2026


