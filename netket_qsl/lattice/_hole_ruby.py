import numpy as np

from ._ruby import Ruby


class HoleRuby(Ruby):
    '''
    Kagome lattice in a Ruby shape as in Semeghini et al. with a hole in the middle. 
    Standard shape is Ruby(a=3.9, extents_down=[3,4,5,6,7,6,5], extents_up=[4,5,6,7,6,5,4]), but we took away the central triangle. 
    
    This lattice is only defined for one particular shape. 
    '''
    
    ## Initilization methods ##
    def __init__(self, a=1.0):
        '''
        Defines the positions of each atom in the lattice
        a : unit cell size in μm
        '''
        super().__init__(a, extents_down=[3,4,5,6,7,6,5], extents_up=[4,5,6,7,6,5,4])


    def check_shape(self, extents_up, extents_down):
        assert extents_down==[3,4,5,6,7,6,5] and extents_up==[4,5,6,7,6,5,4], "This implementation is only implemented for the article's lattice."

        
    def __repr__(self):
        return f'HoleRuby(N={self.N}, shape={self.extents_up, self.extents_down})'

    
    ## Constructing the connexions in the lattice ##
    def construct_vertices(self):
        '''
        Constructs the triangles container, indicating for each triangle which atoms and triangles it possesses
        
        return vertices array (n_vertices,), 
        ex : for vertex #k, vertices[k]['atoms'] = [atoms it possesses], vertices[k]['triangles'] = [triangles it possesses]
        '''
        vertices = super().construct_vertices()

        to_delete = [108, 109, 110]

        # we need to find in which vertices are the atoms to delete

        for k,v in enumerate(vertices):
            for x in to_delete:
                if np.isin(v['atoms'], x ).any():
                    i = np.argmin( np.abs( np.array(v['atoms']) - x)  )
                    del v['atoms'][i]
                    
        for v in vertices:
            for k,x in enumerate(v['atoms']):
                if x>to_delete[2]:
                    v['atoms'][k]-=3

        del_nonbord = []
        for k,v in enumerate(vertices[self.non_border]):
            if len(v['atoms']) != 4:
                del_nonbord.append(k)
        
        self.non_border = np.delete(self.non_border, del_nonbord)

        self.N = self.N - 3

        return vertices


    def construct_positions(self):
        '''
        constructs the positions of the atoms on the latticce
        '''
        super().construct_positions()

        to_delete = [108, 109, 110]

        self.positions = np.delete( self.positions, to_delete, 0 )
 