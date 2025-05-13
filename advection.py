"""
advection.py
A class with functionality to solve the spatially varying 2D advection equation
"""

import numpy as np


class Advection(object):
    """
    A class to solve the spatially varying 2D advection equation
    """

    def __init__(self, **kwargs):
        """
        Initialize the advection solver with velocity fields and grid spacing

        Parameters
        ----------
        kwargs : Dictionary of keyword arguments
        """
        # store kwargs in attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        if "nx" not in self.__dict__:
            self.nx = 64
        if "ny" not in self.__dict__:
            self.ny = 64
        if "Lx" not in self.__dict__:
            self.Lx = 1.0
        if "Ly" not in self.__dict__:
            self.Ly = 1.0
        
        if "kspec" not in self.__dict__:
            self.kspec = 2.0
        if "sigma" not in self.__dict__:
            self.sigma = 1.0

        # options for scalar initial conditions
        if "si_op" not in self.__dict__:
            self.si_op = 0
        
        self._init_grid()
        self._init_velocity_field()
        self._init_scalar()

    def _init_grid(self):
        """
        Initializes the grid for the advection equation
        """
        # Create coordinate arrays
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        
        # Create coordinate arrays
        x = np.linspace(-self.Lx/2, self.Lx/2, self.nx)
        y = np.linspace(-self.Ly/2, self.Ly/2, self.ny)
        self.X, self.Y = np.meshgrid(x, y)  # 2D position arrays
        
    def _init_velocity_field(self):
        """
        Initializes a 2D, periodic gaussian random field with a given power spectrum

        Parameters
        ----------
        kspec : float
            The spectral index of the power spectrum
        sigma : float
            The standard deviation of the velocity field
        """
        # use numpy to generate a 2D, periodic gaussian random field with a given power spectrum
        self.vx = np.zeros((self.nx, self.ny))
        self.vy = np.zeros((self.nx, self.ny))


        kx = np.fft.fftfreq(self.nx, d=self.dx) * 2*np.pi
        ky = np.fft.fftfreq(self.ny, d=self.dy) * 2*np.pi
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX**2 + KY**2).T
        print(K.shape)

        # Generate random phases
        phix = 2*np.pi * np.random.uniform(size=(self.nx,self.ny))
        phiy = 2*np.pi * np.random.uniform(size=(self.nx,self.ny))

        # Create power spectrum in Fourier space
        vx_k = np.zeros((self.nx,self.ny), dtype=complex)
        vy_k = np.zeros((self.nx,self.ny), dtype=complex)
        mask = K > 0
        vx_k[mask] = K[mask]**(-1*self.kspec) * np.exp(1j * phix[mask])
        vy_k[mask] = K[mask]**(-1*self.kspec) * np.exp(1j * phiy[mask])

        # Transform back to real space
        self.vx = np.real(np.fft.ifft2(vx_k))
        self.vy = np.real(np.fft.ifft2(vy_k))

        sigma = np.sqrt(np.std(self.vx)**2 + np.std(self.vy)**2)
        self.vx *= self.sigma/sigma
        self.vy *= self.sigma/sigma

    def _init_scalar(self):
        """
        Initializes the scalar field
        """

        if self.si_op == 0:
            R = np.sqrt((self.X/self.Lx)**2 + (self.Y/self.Ly)**2)
            self.scalar = 1.*(R<0.25)
        else:
            raise ValueError("Invalid scalar initial condition option")

    def solve(self):
        """
        Solve the advection equation
        """
        pass