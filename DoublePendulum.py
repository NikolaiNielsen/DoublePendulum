import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gui

r = 1
r1 = 0.5
r2 = 1-r1

m1 = 1
m2 = 1
g = -10

N = 10000
dt = 0.01


def gen_to_cart(theta1, theta2, r1, r2):
    """
    Calculates the cartesian positions of the masses of the double pendulum,
    from the generalised angle coordiantes. Assumes the pendulum is hung at the
    origin

    inputs:
    - theta1, theta2: floats, generalised coordiantes
    - r1, r2: floats, lengths of the pendulum rods

    outputs:
    - x1: float.  x-pos of first mass
    - y1: float.  y-pos of first mass
    - x2: float.  x-pos of second mass
    - y2: float.  y-pos of second mass
    """

    x1 = r1 * np.sin(theta1)
    y1 = - r1 * np.cos(theta1)

    x2 = x1 + r2 * np.sin(theta2)
    y2 = y1 - r2 * np.cos(theta2)

    return x1, y1, x2, y2


def cart_to_gen(x1, y1, x2, y2, r1, r2):
    theta1 = np.arctan2(x1, y1)
    theta2 = np.arctan2(x2-x1, y2-y1)
    return theta1, theta2


def calc_omega1(theta1, theta2, r1, r2, m1, m2, g):
    """
    Calculates the acceleration of the first pendulum mass
    """

    mass = m2 / (m1 + m2)
    numerator = g/r1 * (mass * np.sin(theta2) * (np.cos(theta1 - theta2) +
                                                 np.sin(theta1 - theta2)) -
                        np.sin(theta1))
    denominator = 1 + mass * np.cos(2 * (theta1 - theta2))

    result = numerator / denominator
    return result


def calc_omega2(f, theta1, theta2, r1, r2, m1, m2, g):
    """
    Calculates the acceleration of the second pendulum mass
    """

    res = (r1/r2 * f * (np.cos(theta1 - theta2) - np.sin(theta1 - theta2)) -
           g/r2 * np.sin(theta2))

    return res


def calc_derivs(state, r1, r2, m1, m2, g):
    """
    Calculate the derivatives of theta1, omega1, theta2 and omega2.
    Ripped from http://scienceworld.wolfram.com/physics/DoublePendulum.html.
    """

    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    delta = state[2] - state[0]
    den1 = (m1+m2) * r1 - m2 * r1 * np.cos(delta) * np.cos(delta)
    dydx[1] = ((m2 * r1 * state[1] * state[1] * np.sin(delta) * np.cos(delta) +
                m2 * g * np.sin(state[2]) * np.cos(delta) +
                m2 * r2 * state[3] * state[3] * np.sin(delta) -
                (m1+m2) * g * np.sin(state[0])) / den1)

    dydx[2] = state[3]

    den2 = (r2/r1) * den1
    dydx[3] = ((- m2*r2*state[3] * state[3] * np.sin(delta) * np.cos(delta) +
               (m1+m2) * g * np.sin(state[0]) * np.cos(delta) -
               (m1+m2) * r1 * state[1] * state[1] * np.sin(delta) -
               (m1+m2) * g * np.sin(state[2])) / den2)

    return dydx


def RK4(state, dt, coeffs):
    """
    Calculates the values of the generalised coordinates at the next time step,
    using the 4th order Runge-Kutta method.
    """

    k1 = dt * calc_derivs(state, *coeffs)
    k2 = dt * calc_derivs(state+k1/2, *coeffs)
    k3 = dt * calc_derivs(state+k2/2, *coeffs)
    k4 = dt * calc_derivs(state+k3, *coeffs)
    state = state + (k1 + 2*k2 + 2*k3 + k4)/6

    return state


def simulate(theta1, theta2, N, dt, r1, r2, m1, m2, g):
    state = np.zeros((N, 4))
    state[0] = [theta1, 0, theta2, 0]
    coeffs = (r1, r2, m1, m2, g)
    for i in range(1, N):
        state[i] = RK4(state[i-1], dt, coeffs)

    dt_arr = np.ones(N) * dt
    T = np.cumsum(dt_arr) - dt

    theta1_arr = state[:, 0]
    theta2_arr = state[:, 2]

    x1, y1, x2, y2 = gen_to_cart(theta1_arr, theta2_arr, r1, r2)
    pot1, pot2 = calc_potential_energy(y1, y2, g, m1, m2)
    kin1, kin2 = calc_kinetic_energy(state, m1, m2, r1, r2)
    tot = calc_total_energy(state, m1, m2, r1, r2, g)

    return x1, y1, x2, y2, T, pot1, pot2, kin1, kin2, tot


def calc_starting_on_mouse(x1, y1, r1, r2):
    """
    From
    https://stackoverflow.com/questions/3349125/circle-circle-intersection-points
    """

    x0 = 0
    y0 = 0
    d = np.sqrt(x1**2 + y1**2)

    # First we check if the point is outside of the combined radius
    if d >= r1+r2:
        # Calculate the angle and new points
        theta = np.arccos(x1/d)
        x3 = r1 * np.cos(theta)
        y3 = r1 * np.sin(theta)
        x1_new = (r1+r2) * np.cos(theta)
        y1_new = (r1+r2) * np.sin(theta)

        # We need to flip the y-coordinates if y1 is below 0
        y1_new = y1_new if y1 > 0 else -y1_new
        y3 = y3 if y1 > 0 else -y3
        return x3, y3, x1_new, y1_new

    a = (r1**2 - r2**2 + d**2)/(2*d)
    h = np.sqrt(r1**2 - a**2)
    x2 = x0 + a*(x1 - x0)/d
    y2 = y0 + a*(y1 - y0)/d
    y31 = y2 - h * (x1-x0)/d
    x31 = x2 + h * (y1-y0)/d

    y32 = y2 + h*(x1-x0)/d
    x32 = x2 - h*(y1-y0)/d

    y31 = -y31
    y32 = -y32
    return (x31, y31, x1, y1) if y31 >= y32 else (x32, y32, x1, y1)


def calc_total_energy(state, m1, m2, r1, r2, g):
    t1, o1, t2, o2 = state.T
    dt = t1-t2
    kin = (0.5*(m1*m2)*r1*r1*o1*o2 + 0.5*m2*r2*r2*o2*o2 +
           m2*r1*r2*o1*o2*np.cos(dt))
    pot = -(m1+m2)*g*r1*np.cos(t1) - m2*g*r2*np.cos(t2)

    tot = kin+pot
    return tot


def calc_potential_energy(y1, y2, g, m1, m2):
    """
    Calculates the potential energy of the system for each step in the
    simulation.
    """
    pot1 = y1 * g * m1
    pot2 = y2 * g * m2
    return pot1, pot2


def calc_kinetic_energy(state, m1, m2, r1, r2):
    """
    Calculates the kinetic energy of the system for each time step
    """
    t1, o1, t2, o2 = state.T
    kin1 = 0.5 * m1 * r1*r1 * o1*o1
    trig = np.cos(t1-t2)
    kin2 = 0.5 * m2 * (r1*r1*o1*o1 + r2*r2*o2*o2 + 2*r1*r2*o1*o2*trig)

    return kin1, kin2


def plot_circle(x0, y0, r, ax):
    N = 50
    theta = np.linspace(0, 2*np.pi, N)
    x = r*np.cos(theta) + x0
    y = r*np.sin(theta) + y0
    ax.plot(x, y)


def animation_window():

    # Size specs. We want a figure of width 6 and height 8
    w, h = 6, 8
    # We want the border to be 0.05*width, so converting that to fractional
    # height is width*0.05/height
    w_border = 0.05
    h_border = (w*w_border)/h
    # Width is just 2-border
    w_1 = 1-2*w_border
    h_1 = (w*w_1)/h
    # Bottom position of ax1. We want the bottom to be at top, minus 1 border,
    # minus the height of the axes
    b_1 = 1 - h_1 - h_border
    w_2 = w_1
    # Height of ax2 is what ever is left
    h_2 = 1-h_1-2*h_border

    fig = plt.figure(figsize=(w, h))
    ax1 = fig.add_axes([w_border, b_1, w_1, h_1])
    ax2 = fig.add_axes([w_border, h_border, w_2, h_2])

    # We don't need tick marks
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    return fig, ax1, ax2


class Animator(object):
    # The animator object, created for every simulation we want to plot
    def __init__(self, fig, ax, ax2, x1, y1, x2, y2, T,
                 pot1, kin1, pot2, kin2, tot):
        # plug stuff into the object and create the empty line
        self.lims = (-1.1, 1.1)
        self.ax = ax
        self.fig = fig
        self.ax2 = ax2
        self.ax.axis('equal')
        self.ax.set_xlim(*self.lims)
        self.ax.set_ylim(*self.lims)
        self.plot, = ax.plot([], [], marker='o', markerfacecolor='r', c='k',
                             linewidth=4, markersize=15)
        naught = np.zeros(x1.size)
        self.xdata = np.array((naught, x1, x2)).T
        self.ydata = np.array((naught, y1, y2)).T
        self.time = T
        self.time_text = self.ax.text(-0.9, 0.9, 'Time: 0')
        self.pot1 = pot1
        self.pot2 = pot2
        self.kin1 = kin1
        self.kin2 = kin2

        self.total_energy = tot
        self.pot1_plot = self.ax2.plot(self.pot1)
        self.pot2_plot = self.ax2.plot(self.pot2)
        self.kin1_plot = self.ax2.plot(self.kin1)
        self.kin2_plot = self.ax2.plot(self.kin2)
        self.total_plot = self.ax2.plot(self.total_energy)

        self.artists = [self.plot, self.time_text]

    def _init(self):
        # function to clear the line every time it is plotted (init function
        # for FuncAnim)
        self.plot.set_data([], [])
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_aspect('equal')
        return self.artists

    def __call__(self, i):
        self.time_text.set_text(f'Time: {self.time[i]:.2f}')
        self.plot.set_data(self.xdata[i], self.ydata[i])
        return self.artists


def _on_mouse(event, r1, r2, ax, ax2, fig, N, dt, m1, m2, g):
    if event.button != 1:
        return

    x0 = event.xdata
    y0 = event.ydata

    if x0 is None or y0 is None:
        # Mouse is out of bounds
        return

    x1, y1, x2, y2 = calc_starting_on_mouse(x0, y0, r1, r2)
    theta1, theta2 = cart_to_gen(x1, y1, x2, y2, r1, r2)
    x1, y1, x2, y2, T, p1, k1, p2, k2, tot = simulate(theta1, theta2, N, dt,
                                                      r1, r2, m1, m2, g)

    A = Animator(fig, ax, ax2, x1, y1, x2, y2, T, p1, k1, p2, k2, tot)

    ani = animation.FuncAnimation(fig, A, frames=N,
                                  init_func=A._init,
                                  save_count=100,
                                  blit=True,
                                  interval=dt*1000,
                                  repeat=False)
    fig.canvas.draw()


if __name__ == '__main__':
    gui.main()
