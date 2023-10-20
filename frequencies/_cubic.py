import numpy as np

from scipy.interpolate import interp1d



def cubic_sweep(sweep_time):
    '''
    Cubic part of the Δ sweep
    sweep_time : total time of the experiment (in μs)
    
    return : function of the sweep such that f(t) = Δ(t) [Mrad/μs] and the fit coefficients (4,)
    '''
    
    # find the polynomial so we can go further than sweep_time
    # these choices are the most used ones, so we saved the coefficients
    if sweep_time == 2.5:
        f = np.array([52.2811409 , -245.24908958,  385.23213891, -186.70318457]) #polynomial coefficients (in rad/s) for sweep_time=2.5 
    
    elif sweep_time==25:
        f = np.array([ 5.22811409e-02, -2.45249090e+00,  3.85232139e+01, -1.86703185e+02]) ##polynomial coefficients (in rad/s) for sweep_time=25 
    
    # for another sweep_time, we have to find the sweep from scratch
    else:
        tTotal = sweep_time*4/5
        tInf = sweep_time*4/5/2
        fMin = -8
        fMax = 9.4
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
        
        

    

class Frequencies:
    '''
    Defines the frequencies schedules Ω(t) and Δ(t) for a given sweep_time.
    Allows to get both frequencies at the time we want. The functions are vectoized and can be used outside the time range [0,sweep_time].
    It is also possible to integrate and derivate the frequencies over any time interval. 
    '''
    def __repr__(self):
        return f'Ω,Δ(T={self.sweep_time})'
    
    def __init__(self, sweep_time):
        '''
        sweep_time : total time of the experiment (in μs)
                     This corresponds to the time where Δ/2π = 9.6
        '''
        self.sweep_time = sweep_time
        self.sweep, self.f = cubic_sweep(sweep_time)
        
        self.Ωf = self.Ω(sweep_time)
        
    def Ω(self, t):
        '''
        Defines the default Ω schedule for the time evolution over a total sweep_time
        t : time of the evolution (in μs)

        return : corresponding Ω (in Mrad/μs)
        '''
        flag = (t < self.sweep_time/5)
        return 2.8*t *2*np.pi/self.sweep_time*2.5*flag + 1.4*2*np.pi*(1-flag)


    def Δ(self, t):
        '''
        Defines the default Δ schedule for the time evolution over a total sweep_time
        t : time of the evolution (in μs)

        return : corresponding Δ (in Mrad/μs)
        '''
        flag = (t < self.sweep_time/5)
        return -2*np.pi*8*(flag) + (1 - flag)*self.sweep(t)
        
        
 
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
                return 2*np.pi*1.4 * (t2+t1)/2/T
            else:
                return 2*np.pi*1.4 * (t2 - 0.5*T - 0.5*t1**2/T)/(t2-t1)
        else:
            return 2*np.pi*1.4
        
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
                return (F(t2)-F(T) - 2*np.pi*8*(T-t1))/(t2-t1)
        else:
            return -2*np.pi*8
    
