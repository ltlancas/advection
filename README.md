# advection

This small repository provides a solver for the 2-dimensional advection equation with a
spatially varying velocity field using a finite volume method.

It can be run easily with the following code

```
from advection import Advection

adv = Advection()
```

this initializes the solver with default settings which is a 64 by 64 grid with a
gaussian random velocity field with unit rms. The scalar field is initialized as 
unity within a sphere at the center of the domain and zero elsewhere. Default
evolution is done to second order accuracy in space and time.

The solution is then evolved forward for time `t` as `adv.solve(t)`
