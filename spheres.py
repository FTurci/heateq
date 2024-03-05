# Francesco Turci 2023
# f.turci@bristol.ac.uk
# This code is licensed under GPL-3.0-or-later

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class HotSpheres:
    """
    A class to simulate heat diffusion with hot spheres in a 2D grid.

    Attributes:
        N (int): Number of hot spheres.
        w (float): Width of the grid.
        h (float): Height of the grid.
        r (float): Radius of hot spheres.
        D (float): Diffusion coefficient.
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.
        Tcold (float): Initial temperature of the grid.
        Thot (float): Temperature of the hot spheres.

    Methods:
        do_timestep(self, u0, u): Perform a time step of the heat diffusion simulation.
        update(self, frame): Update the plot for each animation frame.
        display(self, nsteps=100): Display the heat diffusion animation.
    """

    def __init__(
        self,
        N=16,
        w=10.0,
        h=10.0,
        r=0.1,
        D=1.0,
        dx=0.05,
        dy=0.05,
        Tcold=300.0,
        Thot=700.0,
    ):
        # Initialize class attributes
        self.Tcold = Tcold
        self.Thot = Thot
        self.D = D

        nx, ny = int(w / dx), int(h / dy)

        self.dx2, self.dy2 = dx * dx, dy * dy
        dt = self.dx2 * self.dy2 / (2 * D * (self.dx2 + self.dy2))

        self.dt = dt

        u0 = Tcold * np.ones((nx, ny))

        X = np.random.uniform(0, w, size=N)
        Y = np.random.uniform(0, w, size=N)

        # Initial conditions - circles of radius r centered at (cx, cy) (mm)
        for cx, cy in zip(X, Y):
            r2 = r**2
            for i in range(nx):
                for j in range(ny):
                    p2 = (i * dx - cx) ** 2 + (j * dy - cy) ** 2
                    if p2 < r2:
                        u0[i, j] = Thot

        # Apply periodic boundary conditions for the x-axis (left and right edges)
        u0[0, :] = u0[-2, :]
        u0[-1, :] = u0[1, :]

        # Apply periodic boundary conditions for the y-axis (top and bottom edges)
        u0[:, 0] = u0[:, -2]
        u0[:, -1] = u0[:, 1]

        self.u0 = u0
        self.u = u0.copy()

    def do_timestep(self, u0, u):
        D = self.D
        dt = self.dt
        # Propagate with forward-difference in time, central-difference in space
        u[1:-1, 1:-1] = u0[1:-1, 1:-1] + D * dt * (
            (u0[2:, 1:-1] - 2 * u0[1:-1, 1:-1] + u0[:-2, 1:-1]) / self.dx2
            + (u0[1:-1, 2:] - 2 * u0[1:-1, 1:-1] + u0[1:-1, :-2]) / self.dy2
        )

        # Apply periodic boundary conditions for the x-axis (left and right edges)
        u[0, :] = u[-2, :]
        u[-1, :] = u[1, :]

        # Apply periodic boundary conditions for the y-axis (top and bottom edges)
        u[:, 0] = u[:, -2]
        u[:, -1] = u[:, 1]

        u0 = u.copy()
        return u0, u

    def update(self, frame):
        self.u0, self.u = self.do_timestep(self.u0, self.u)
        self.im.set_array(self.u[1:-1, 1:-1])
        return (self.im,)

    def display(self, nsteps=1000):
        # Create a figure
        self.fig, self.ax = plt.subplots(figsize=(3, 3))

        # Initialize the imshow plot
        self.im = self.ax.imshow(
            self.u[1:-1, 1:-1],
            cmap=plt.get_cmap("hot"),
            vmin=self.Tcold,
            vmax=self.Thot,
        )
        self.ax.set_axis_off()

        # Create the animation
        self.ani = FuncAnimation(
            self.fig,
            func=self.update,
            frames=range(nsteps),
            interval=1,
            blit=True,
            repeat=False,
        )
        plt.tight_layout()
    
