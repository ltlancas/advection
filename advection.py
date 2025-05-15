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
        if "vel_op" not in self.__dict__:
            self.vel_op = 0
        if "kmax" not in self.__dict__:
            self.kmax = 0.0

        # options for scalar initial conditions
        if "si_op" not in self.__dict__:
            self.si_op = 0
        
        self._init_grid()
        self._init_velocity_field()
        self._init_scalar()

        if "tlim" not in self.__dict__:
            self.tlim = 10*max(self.Lx/self.sigma, self.Ly/self.sigma)
        if "cfl" not in self.__dict__:
            self.cfl = 0.5
        self.dt = self.cfl*min(self.dx/np.max(np.abs(self.vx)), self.dy/np.max(np.abs(self.vy)))
        self.nt = int(self.tlim/self.dt)
        if "x_order" not in self.__dict__:
            self.x_order = 2
        if "y_order" not in self.__dict__:
            self.y_order = 2

    def _init_grid(self):
        """
        Initializes the grid for the advection equation
        """
        # Create coordinate arrays
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        
        # Create coordinate arrays
        x = np.linspace(-self.Lx/2+self.dx/2, self.Lx/2-self.dx/2, self.nx)
        y = np.linspace(-self.Ly/2+self.dy/2, self.Ly/2+self.dy/2, self.ny)
        X, Y = np.meshgrid(x, y)  # 2D position arrays
        (self.X,self.Y) = (X.T, Y.T)
        
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
        #self.vx = np.zeros((self.nx, self.ny))
        #self.vy = np.zeros((self.nx, self.ny))

        if self.vel_op == 0:
            self.vx = np.ones((self.nx, self.ny))
            if "vx_scale" in self.__dict__:
                self.vx *= self.vx_scale
            self.vy = np.ones((self.nx, self.ny))
            if "vy_scale" in self.__dict__:
                self.vy *= self.vy_scale
        elif self.vel_op >= 1:
            kx = np.fft.fftfreq(self.nx, d=self.dx) * 2*np.pi
            ky = np.fft.fftfreq(self.ny, d=self.dy) * 2*np.pi
            KX, KY = np.meshgrid(kx, ky)
            (KX,KY) = (KX.T, KY.T)
            K = np.sqrt(KX**2 + KY**2)
            (self.KX,self.KY) = (KX, KY)

            # Generate random phases in a way that guarantees a real field
            phix = self._get_random_phase()
            phiy = self._get_random_phase()
            # Create power spectrum in Fourier space
            vx_k = np.zeros((self.nx,self.ny), dtype=complex)
            vy_k = np.zeros((self.nx,self.ny), dtype=complex)
            mask = K > 0
            vx_k[mask] = K[mask]**(-0.5*self.kspec) * np.exp(1j * phix[mask])
            vy_k[mask] = K[mask]**(-0.5*self.kspec) * np.exp(1j * phiy[mask])

            # calculate solenoidal field
            Az_k = np.zeros((self.nx,self.ny), dtype=complex)
            Az_k[mask] = 1.j*(KX*vy_k - KY*vx_k)[mask]/(K[mask]**2)
            Az_k[self.nx//2] = 0
            Az_k[:,self.ny//2] = 0
            Az = np.fft.ifftn(Az_k).real
            # minus sign in principle comes from the tranformation
            # it's not strictly necessary to do this
            vx_sol = (np.roll(Az,-1,axis=1) - np.roll(Az,1,axis=1))/(2*self.dy)
            vy_sol = -1*(np.roll(Az,-1,axis=0) - np.roll(Az,1,axis=0))/(2*self.dx)

            # calculate compressive field
            v_k_comp = np.zeros((self.nx,self.ny), dtype=np.complex128)
            v_k_comp[mask] = 1.j*(KX*vx_k + KY*vy_k)[mask]/(K[mask]**2)
            v_k_comp[self.nx//2] = 0
            v_k_comp[:,self.ny//2] = 0
            v_comp = np.fft.ifftn(v_k_comp).real
            vx_comp = -1*(np.roll(v_comp,-1,axis=0) - np.roll(v_comp,1,axis=0))/(2*self.dx)
            vy_comp = -1*(np.roll(v_comp,-1,axis=1) - np.roll(v_comp,1,axis=1))/(2*self.dy)

            if self.vel_op == 1:
                # keep entire field, compressive and solenoidal parts
                self.vx = vx_comp + vx_sol
                self.vy = vy_comp + vy_sol
            elif self.vel_op == 2:
                # return only the solenoidal field
                self.vx = vx_sol
                self.vy = vy_sol
            elif self.vel_op == 3:
                # return only the compressive field
                self.vx = vx_comp
                self.vy = vy_comp
            else:
                raise ValueError("Invalid velocity field option")
           

            sigma = np.sqrt(np.std(self.vx)**2 + np.std(self.vy)**2)
            self.vx *= self.sigma/sigma
            self.vy *= self.sigma/sigma
        else:
            raise ValueError("Invalid velocity field option")

        # get interface velocities
        self.vx_int = 0.5*(self.vx + np.roll(self.vx,1,axis=1))
        self.vy_int = 0.5*(self.vy + np.roll(self.vy,1,axis=0))

    def _get_random_phase(self):
        # creates random phase matrix in a way that
        # guarantees that the fourier transform is real
        phi = 2*np.pi * np.random.uniform(size=(self.nx,self.ny))
        # first make sure maximum and zero frequency phases are zero
        phi[0,0] = 0
        phi[self.nx//2] = 0
        phi[:,self.ny//2] = 0
        # make anti-symmetric
        for i in range(1,self.nx//2):
            phi[i,0] = -1*phi[-i,0]
            for j in range(1,self.ny//2):
                phi[i,j] = -1*phi[-i,-j]
                phi[-i,j] = -1*phi[i,-j]
        for j in range(1,self.ny//2):
            phi[0,j] = -1*phi[0,-j]
        return phi


    def _init_scalar(self):
        """
        Initializes the scalar field
        """

        if self.si_op == 0:
            R = np.sqrt((self.X/self.Lx)**2 + (self.Y/self.Ly)**2)
            self.scalar = 1.*(R<0.25)
        else:
            raise ValueError("Invalid scalar initial condition option")

    def solve(self, T):
        """
        Solve the advection equation for a given time step
        """
        tev = 0
        while tev < T:
            if tev + self.dt > T:
                dt = T-tev
            else:
                dt = self.dt
            self.single_iteration(dt)
            tev += dt

    def single_iteration(self, dt):
        """
        Perform a single iteration of the advection equation
        """
        # use predictor corrector scheme if order > 1
        if (self.x_order > 1 or self.y_order > 1):
            # 2nd-order integratoin
            self.reconstrcut_x(self.scalar)
            self.reconstrcut_y(self.scalar)

            # construct fluxes
            F = -0.5*(dt/self.dx)*self.s_intx*self.vx_int
            G = -0.5*(dt/self.dy)*self.s_inty*self.vy_int

            # calculate flux divergences
            Fdiff = np.roll(F,-1,axis=1) - F
            Gdiff = np.roll(G,-1,axis=0) - G
            # take half step forward
            sdt2 = self.scalar + (Fdiff + Gdiff)

            self.reconstrcut_x(sdt2)
            self.reconstrcut_y(sdt2)

            # construct fluxes
            F = -1*(dt/self.dx)*self.s_intx*self.vx_int
            G = -1*(dt/self.dy)*self.s_inty*self.vy_int

            # calculate flux divergences
            Fdiff = np.roll(F,-1,axis=1) - F
            Gdiff = np.roll(G,-1,axis=0) - G
            self.scalar += Fdiff + Gdiff
        else:
            # perform reconstruction
            self.reconstrcut_x(self.scalar)
            self.reconstrcut_y(self.scalar)

            # construct fluxes
            F = -1*(dt/self.dx)*self.s_intx*self.vx_int
            G = -1*(dt/self.dy)*self.s_inty*self.vy_int

            # calculate flux divergences
            Fdiff = np.roll(F,-1,axis=1) - F
            Gdiff = np.roll(G,-1,axis=0) - G

            # apply flux divergences to the scalar field
            self.scalar += Fdiff + Gdiff

    def reconstrcut_x(self, s):
        """
        Technically these are both the reconstructions and the 'Riemann' solvers
        """
        # first reconstruct, either first or second order
        if self.x_order == 1:
            (sr,sl) = (s,np.roll(s,1,axis=1))
        elif self.x_order == 2:
            dsL = s - np.roll(s,1,axis=1)
            dsR = np.roll(s,-1,axis=1) - s
            dsC = (np.roll(s,-1,axis=1) - np.roll(s,1,axis=1))/2
            ds = np.sign(dsC)*np.minimum(2*np.minimum(np.abs(dsL),np.abs(dsR)), np.abs(dsC))
            sr = s - ds/2
            sl = np.roll(s + ds/2, 1,axis=1)
        else:
            raise ValueError("Invalid x-order")

        # then apply the "Riemann solver" based on velocity
        self.s_intx = sr*(self.vx_int < 0) + sl*(self.vx_int > 0) + 0.5*(sr+sl)*(self.vx_int == 0)

    def reconstrcut_y(self, s):
        # first reconstruct, either first or second order
        if self.y_order == 1:
            (sr,sl) = (s,np.roll(s,1,axis=0))
        elif self.y_order == 2:
            dsL = s - np.roll(s,1,axis=0)
            dsR = np.roll(s,-1,axis=0) - s
            dsC = (np.roll(s,-1,axis=0) - np.roll(s,1,axis=0))/2
            ds = np.sign(dsC)*np.minimum(2*np.minimum(np.abs(dsL),np.abs(dsR)), np.abs(dsC))
            sr = s - ds/2
            sl = np.roll(s + ds/2, 1,axis=0)
        else:
            raise ValueError("Invalid y-order")

        # then apply the "Riemann solver" based on velocity
        self.s_inty = sr*(self.vy_int < 0) + sl*(self.vy_int > 0) + 0.5*(sr+sl)*(self.vy_int == 0)