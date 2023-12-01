import abc
import numpy as np

class Frequency:
    '''
    Defines the frequencies schedules Ω(t) and Δ(t) for a given sweep_time.
    Allows to get both frequencies at the time we want. The functions are vectorized and can be used outside the time range [0,sweep_time].
    It is also possible to integrate and derivate the frequencies over any time interval. 

    The general aspect of the schedule is as follows :
    Omega(t) :
            - [0,sweep_time/5] : linear increase from 0 to Omega
            - [sweep_time/5,infty] : constant at Omega
    Delta(t) :
            - [0,sweep_time/5] : constant at fMin
            - [sweep_time/5,infty] : increase according to chosen subclass
    '''    
    def __init__(self, sweep_time, Omega=1.4):
        '''
        sweep_time : total time of the experiment (in μs)
        Omega : final constant value of Omega(t>sweep_time/5)
        '''
        self.sweep_time = sweep_time
        
        self.Ωf = Omega*2*np.pi
    
    def __repr__(self):
        return f'T={self.sweep_time}, in [{0},{self.Ωf/2/np.pi}],'
        
    def Ω(self, t):
        '''
        Defines the default Ω schedule for the time evolution over a total sweep_time
        t : time of the evolution (in μs)

        return : corresponding Ω (in Mrad/μs)
        '''
        flag = (t < self.sweep_time/5)
        return 5*self.Ωf*t/self.sweep_time*flag + self.Ωf*(1-flag)

    @abc.abstractmethod
    def Δ(self, t):
        '''
        Defines the default Δ schedule for the time evolution over a total sweep_time
        t : time of the evolution (in μs)

        return : corresponding Δ (in Mrad/μs)
        '''
        
 
    def __call__(self, t):
        '''
        Computes both frequencies at time t
        t : time of the evolution (in μs)

        return : corresponding Ω,Δ (in Mrad/μs)        
        '''
        return self.Ω(t), self.Δ(t)
        
        
    def relative(self, t):
        '''
        Computes both frequencies at time t relative to the final Ω
        t : time of the evolution (in μs)

        return : corresponding Ω/Ωf,Δ/Ωf (no unity)        
        '''   
        return self.Ω(t)/self.Ωf, self.Δ(t)/self.Ωf
    
    def herz(self, t):
        '''
        Computes both frequencies at time t in MHz
        t : time of the evolution (in μs)

        return : corresponding Ω/2π,Δ/2π (MHz)        
        '''
        return self.Ω(t)/2/np.pi, self.Δ(t)/2/np.pi

    def mean_Omega(self, t1, t2):
        r'''
        Does the operation $\int_t1^t2 Ω(τ) dτ$
        t1 : smallest time (μs)
        t2 : highest time (μs)

        returns : mean of Ω (rad)
        '''
        T = self.sweep_time/5
        if t1 < T:
            if t2 < T:
                return self.Ωf * (t2+t1)/2/T
            else:
                return self.Ωf * (t2 - 0.5*T - 0.5*t1**2/T)/(t2-t1)
        else:
            return self.Ωf
        
    @abc.abstractmethod
    def mean_Delta(self, t1, t2):
        r'''
        Does the operation $\int_t1^t2 Δ(τ) dτ$
        t1 : smallest time (μs)
        t2 : highest time (μs)

        returns : mean of Δ (rad)
        '''
