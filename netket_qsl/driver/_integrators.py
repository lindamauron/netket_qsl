import numpy as np
import abc

class Integrator:
    def __init__(self,dt):
        self.h = dt

    @abc.abstractmethod
    def __call__(self,driver,y):
        """
        integrates the y paremeters for a derivative given by driver._backward 
        """




class RK4(Integrator):
    def __init__(self,dt:float):
        return super().__init__(dt)
    def __repr__(self):
        return f"RK4(dt={self.h})"
    
    def __call__(self,driver,y):
        k1 = self.h*driver._backward(driver.t,y)
        k2 = self.h*driver._backward(driver.t+self.h/2, y+k1/2)
        k3 = self.h*driver._backward(driver.t+self.h/2, y+k2/2)
        k4 = self.h*driver._backward(driver.t+self.h, y+k3)

        y = y+(k1+2*k2+2*k3+k4)/6

        #renormalize things
        norm = np.linalg.norm(y, axis=-1)
        y = y.at[:,0].set( y[:,0]/norm )
        y = y.at[:,1].set( y[:,1]/norm )

        if np.isnan(y).any() or np.isinf(y).any():
            return False, None

        return True, y

