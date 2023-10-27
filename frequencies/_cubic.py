import numpy as np

from scipy.interpolate import interp1d

from .base import Frequency

def cubic_sweep(sweep_time, fMin = -8, fMax = 9.4):
    '''
    Cubic part of the Δ sweep
    sweep_time : total time of the experiment (in μs)
    
    return : function of the sweep such that f(t) = Δ(t) [Mrad/μs] and the fit coefficients (4,)
    '''
    
    # find the polynomial so we can go further than sweep_time
    # these choices are the most used ones, so we saved the coefficients
    if sweep_time == 2.5 and fMin==-8 and fMax==9.4:
        f = np.array([52.2811409 , -245.24908958,  385.23213891, -186.70318457]) #polynomial coefficients (in rad/s) for sweep_time=2.5 
    
    elif sweep_time==25 and fMin==-8 and fMax==9.4:
        f = np.array([ 5.22811409e-02, -2.45249090e+00,  3.85232139e+01, -1.86703185e+02]) ##polynomial coefficients (in rad/s) for sweep_time=25 
    
    # for another sweep_time, we have to find the sweep from scratch
    else:
        tTotal = sweep_time*4/5
        tInf = sweep_time*4/5/2
        #fMin = -8
        #fMax = 9.4
        fInf = 2.4
        slope = 0.4/sweep_time*2.5
        midT = 0.05*sweep_time/2.5

        midF = slope * midT
        tPoints1 = np.array([0., tInf-midT, tInf, tInf+midT, tTotal])
        fPoints1 = np.array([fMin, fInf-midF, fInf, fInf+midF, fMax])
        fInterp1 = interp1d(tPoints1, fPoints1, kind='cubic',
                            bounds_error=False, fill_value=(fMin, fMax)
                           )
        
        times = np.linspace(sweep_time/5, sweep_time, 10000)
        deltas = 2*np.pi*fInterp1(times-sweep_time/5)

        f = np.polyfit(times, deltas, 3)

        
    return lambda t : (f[0]*t**3 + f[1]*t**2 + f[2]*t + f[3]), f
        
        

    

class Cubic(Frequency):
    '''
    Defines the cubic frequencies schedules Ω(t) and Δ(t) for a given sweep_time.
    Allows to get both frequencies at the time we want. The functions are vectorized and can be used outside the time range [0,sweep_time].
    It is also possible to integrate and derivate the frequencies over any time interval. 

    The general aspect of the schedule is as follows :
    Omega(t) :
            - [0,sweep_time/5] : linear increase from 0 to Omega
            - [sweep_time/5,infty] : constant at Omega
    Delta(t) :
            - [0,sweep_time/5] : constant at fMin
            - [sweep_time/5,infty] : cubic increase
    '''   
    
    def __init__(self, sweep_time, Omega=1.4, fMin=-8, fMax=9.4):
        '''
        sweep_time : total time of the experiment (in μs)
        Omega : final constant value of Omega(t>sweep_time/5)
        fMin : minimal (initial) value of Delta
        fMax : maxima (final) value of Delta
        '''
        super().__init__(sweep_time, Omega)
        self.sweep, self.f = cubic_sweep(sweep_time,fMin,fMax)
        self.Dmin = 2*np.pi*fMin
        self.Dmax = 2*np.pi*fMax

    def __repr__(self):
        return 'CubicSweep(' + super().__repr__()+ f'[{self.Dmin/2/np.pi},{self.Dmax/2/np.pi}])'

    def Δ(self, t):
        '''
        Defines the default Δ schedule for the time evolution over a total sweep_time
        t : time of the evolution (in μs)

        return : corresponding Δ (in Mrad/μs)
        '''
        flag = (t < self.sweep_time/5)
        return self.Dmin*(flag) + (1 - flag)*self.sweep(t)
    
        
    def mean_Delta(self, t1, t2):
        r'''
        Does the operation $\int_t1^t2 Δ(τ) dτ$
        t1 : smallest time (μs)
        t2 : highest time (μs)

        returns : mean of Δ (rad)
        '''
        
        T = self.sweep_time/5
        
        if t2 > T:
            def F(t):
                return self.f[0]/4*t**4 + self.f[1]/3*t**3 + self.f[2]/2*t**2 + self.f[3]*t
            if t1 > T:
                return (F(t2)-F(t1) )/(t2-t1)
            else:
                return (F(t2)-F(T) + self.Dmin*(T-t1))/(t2-t1)
        else:
            return self.Dmin
    

def linear_sweep(sweep_time,fMin,fMax):
    slope = (fMax-fMin)/(4*sweep_time/5)
    offset = fMin - slope*sweep_time/5
    f = 2*np.pi*np.array([slope,offset])

    return lambda t : (f[0]*t+f[1]), f



class Linear(Frequency):
    '''
    Defines the cubic frequencies schedules Ω(t) and Δ(t) for a given sweep_time.
    Allows to get both frequencies at the time we want. The functions are vectorized and can be used outside the time range [0,sweep_time].
    It is also possible to integrate and derivate the frequencies over any time interval. 

    The general aspect of the schedule is as follows :
    Omega(t) :
            - [0,sweep_time/5] : linear increase from 0 to Omega
            - [sweep_time/5,infty] : constant at Omega
    Delta(t) :
            - [0,sweep_time/5] : constant at fMin
            - [sweep_time/5,infty] : linear increase
    ''' 
    
    def __init__(self, sweep_time,Omega=1.4,fMin=-8,fMax=9.4):
        '''
        sweep_time : total time of the experiment (in μs)
        Omega : final constant value of Omega(t>sweep_time/5)
        fMin : minimal (initial) value of Delta
        fMax : maxima (final) value of Delta
        '''
        super().__init__(sweep_time,Omega)
        self.sweep, self.f = linear_sweep(sweep_time,fMin,fMax)
        self.Dmin = 2*np.pi*fMin
        self.Dmax = 2*np.pi*fMax

    
    def __repr__(self):
        return 'Linear sweep(' + super().__repr__()+ ')'

    def Δ(self, t):
        '''
        Defines the default Δ schedule for the time evolution over a total sweep_time
        t : time of the evolution (in μs)

        return : corresponding Δ (in Mrad/μs)
        '''
        flag = (t < self.sweep_time/5)
        return self.Dmin*(flag) + (1 - flag)*self.sweep(t)
        

    def mean_Delta(self, t1, t2):
        r'''
        Does the operation $\int_t1^t2 Δ(τ) dτ$
        t1 : smallest time (μs)
        t2 : highest time (μs)

        returns : mean of Δ (rad)
        '''
        
        T = self.sweep_time/5
        
        if t2 > T:
            def F(t):
                return self.f[0]/2*t**2 + self.f[1]*t
            if t1 > T:
                return (F(t2)-F(t1) )/(t2-t1)
            else:
                return (F(t2)-F(T) + self.Dmin*(T-t1))/(t2-t1)
        else:
            return self.Dmin
    

