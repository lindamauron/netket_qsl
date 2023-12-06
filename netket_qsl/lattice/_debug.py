import numpy as np


from ._square import Square


class OneTriangle(Square):
    '''
    Non peridic Kagome lattice with only one triangle, i.e. 3 sites. 
    This is a special case useful for debugging since the corresponding hilbert space is small nough for all computations. 
    Yet, the defintiion exactly corresponds to the one from the super class, so we just define here for simplicity, yet do not change things. 
    '''
    def __init__(self,a=1.0):
        '''
        a : minimal distance between two atoms
        '''
        return super().__init__(a, 1, 1)
    
    @property
    def hexagons(self):
        return None
    

class TwoTriangles(Square):
    '''
    Non periodic Kagome lattice with only two triangles, i.e. 6 sites.
    This is a special case useful for debugging since the corresponding hilbert space is small nough for all computations. 
    We define the positions of the triangles differently from the other lattice, which is why we do a special case for this
    '''
    def __init__(self,a=1.0):
        '''
        a : minimal distance between two atoms
        '''    

        super().__init__(a,1,2)
    
    def construct_pos(self):
        '''
        constructs the positions of the atoms on the lattice
        
        return : container with all atoms positions (N,2)
        '''
        sq = np.sqrt(3)
        pos = self.a*np.array(
            [np.array([0,0]), 
             np.array([-1/2, sq/2]), 
             np.array([1/2, sq/2]), 
             np.array([-1/2, 3*sq/2]), 
             np.array([1/2, 3*sq/2]), 
             np.array([0, 2*sq])]
                               )
        
        pos = pos - self.a*np.array([1.5, -sq/2])
        return pos
    
    def construct_vertices(self):
        '''
        Constructs the vertices container, indicating for each vertex which atoms and triangles it possesses

        return vertices array (n_vertices,), 
        ex : for vertex #k, vertices[k]['atoms'] = [atoms it possesses], vertices[k]['triangles'] = [triangles it possesses]
        '''

        self.n_vertices = 5
        self.non_border = np.array([])

        atoms = [[1,2,3,4], [0,1], [0,2], [3,5], [4,5]]
        triangles = [[0,1], [0], [0], [1], [1]]
        return np.array([ {'atoms':[int(x) for x in atoms[k]], 'triangles':[int(v) for v in triangles[k] ]} for k in range(self.n_vertices) ])
     
    @property
    def hexagons(self):
        return None
    
    def plot_lattice(self, ax, annotate=False, plot_lines=False, sample=None):
        '''
        Plots the lattice on a chosen figure with annotation of the atoms
        For example : 

        fig, ax = plt.subplots(1,1, figsize=(6,4))
        lattice.plot_lattice(ax, True, True)
        ax.axis('off')
        plt.show() 


        ax : AxesSubPlot on which to plot the lattice
        annotate : boolean indicating wether we write down the indices of the atoms
        plot_lines : boolean indicating wether we want the lattice structure (lines) plotted as well
                     (helps for visibility but takes quite longer to load, i.e. do not do many times)
        sample : one specific sample to draw. In this sample, +1 is red and -1 is green
        '''
        super().plot_lattice(ax, annotate=annotate, plot_lines=plot_lines, sample=sample)
            
        # If indicated, plot the lattice's structure    
        if plot_lines:
            pos = self.positions/self.a

            ax.plot( pos.T[0,:3], pos.T[1,:3], color='k', linestyle='-')
            ax.plot( pos.T[0,3:], pos.T[1,3:], color='k', linestyle='-')
            
            ax.plot( [-1.5,-1], [np.sqrt(3)/2, np.sqrt(3)], color='k', linestyle='-')
            ax.plot( [-2,-1.5], [2*np.sqrt(3), 5*np.sqrt(3)/2], color='k', linestyle='-')
            
            
            ax.set_xlim([-3,pos[-1,0]+2])
            ax.set_ylim([-3,pos[-1,1]+3])

        return
    