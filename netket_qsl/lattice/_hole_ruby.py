import numpy as np

from ._ruby import Ruby
from ._hexagons import Hexagons

class HoleRuby(Ruby):
    '''
    Kagome lattice in a Ruby shape as in Semeghini et al. with a hole in the middle. 
    Standard shape is Ruby(a=3.9, extents_down=[3,4,5,6,7,6,5], extents_up=[4,5,6,7,6,5,4]), but we took away the central triangle. 
    
    This lattice is only defined for two particular shapes. 
    '''
    
    ## Initilization methods ##
    def __init__(self, a, N=None, extents_down=None, extents_up=None, to_delete=None):
        '''
        Defines the positions of each atom in the lattice
        a : unit cell size in μm
        N: umber of sites (choose between 216 or 285 for now)
        extents_down : list of extents of the down triangles
        extents_up : list of extents of the up triangles
                        Both should be compatible
        to_delete: sites of the lattice to delete, i..e where the hole is located
                    the sites should cover a whole triangle (for now, only compatible with one single triangle)
        '''
        if N==216:
            self.to_delete = np.sort([108, 109, 110])
            super().__init__(a, extents_down=[3,4,5,6,7,6,5], extents_up=[4,5,6,7,6,5,4])

        elif N==285:
            self.to_delete = np.sort([108,109,110])
            super().__init__(a, extents_down=[4,5,6,7,8,7,6,5], extents_up=[5,6,7,8,7,6,5,4])

        elif extents_down and extents_up and to_delete:
            self.to_delete = np.sort(to_delete)
            t = self.to_delete[0]//3
            if not len(to_delete)==3 or (to_delete!=3*t+np.array([0,1,2])).any():
                raise AttributeError(f'One needs to delete a triangle, instead got {self.to_delete}')
            super().__init__(a, extents_down=extents_down, extents_up=extents_up)

        else:
            raise AttributeError(f'There is not implementation with N={N} yet.')


    # def check_shape(self, extents_up, extents_down):
    #     '''
    #     Verifies whether the chosen shape has been implemented.
    #     '''
    #     if not (extents_down==[3,4,5,6,7,6,5] and extents_up==[4,5,6,7,6,5,4]) and not (extents_down==[4,5,6,7,8,7,6,5] and extents_up==[5,6,7,8,7,6,5,4]):
    #         print(extents_down, extents_up)
    #         raise AttributeError("This implementation is only implemented for specific lattices.")

        
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
        to_delete = self.to_delete

        # we need to find in which vertices are the atoms to delete
        for k,v in enumerate(vertices):
            for x in to_delete:
                if np.isin(v['atoms'], x ).any():
                    i = np.argmin( np.abs( np.array(v['atoms']) - x)  )
                    del v['atoms'][i]
                    
        for v in vertices:
            for k,x in enumerate(v['atoms']):
                if x>np.max(to_delete):
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

        to_delete = self.to_delete

        self.positions = np.delete( self.positions, to_delete, 0 )
 
    @property
    def hexagons(self):
        '''
        Defines the (partial) hexagons of the lattice
        
        In this case, to allow for completion in 6-tuples, the hexagons touching the 
        center hole had to be split it two separate contours. 
        Since they touch a border, they are already not closed contours, 
        so this should not matter too much.,
        '''
        if not self._hexagons:
            hexs = super().hexagons

            # we get the hexagons from the lattice without hole
            # hexs = super().hexagons.sites
            to_delete = self.to_delete

            sites = []
            for h in hexs.sites:
                new_hex = []
                for i in h:
                    # before the whole : numerotation is unchanged
                    if i < np.min(to_delete):
                        new_hex.append(i)
                    # the hole is deleted
                    # elif np.isin(i,to_delete):
                        # do nothing
                    # after the hole, numbering has to be updated
                    elif i > np.max(to_delete):
                        new_hex.append( i-len(to_delete) )
                
                sites.append( new_hex )

            # sites = [
            #     [0,9],
            #     [3,12,1],
            #     [6,15,4],
            #     [18,7],
            #     [10,21,33],
            #     [2,13,24,36,22,11],
            #     [5,16,27,39,25,14],
            #     [8,19,30,42,28,17],
            #     [45,31,20],
            #     [34,48,63],
            #     [23,37,51,66,49,35],
            #     [26,40,54,69,52,38],
            #     [29,43,57,72,55,41],
            #     [32,46,60,75,58,44],
            #     [78,61,47],
            #     [64,81,99],
            #     [50,67,84,102,82,65],
            #     [53,70,87,105,85,68],
            #     [56,73,90],
            #     [88,71],
            #     [59,76,93,108,91,74],
            #     [62,79,96,111,94,77],
            #     [114,97,80],
            #     [100,117],
            #     [83,103,120,138,118,101],
            #     [86,106,123,141,121,104],
            #     [144,126,124],
            #     [89,107],
            #     [92,109,129],
            #     [147,127],
            #     [95,112,132,150,130,110],
            #     [98,115,135,153,133,113],
            #     [136,116],
            #     [119,139,156],
            #     [122,142,159,174,157,140],
            #     [125,145,162,177,160,143],
            #     [128,148,165,180,163,146],
            #     [131,151,168,183,166,149],
            #     [134,154,171,186,169,152],
            #     [137,172,155],
            #     [158,175,189],
            #     [161,178,192,204,190,176],
            #     [164,181,195,207,193,179],
            #     [167,184,198,210,196,182],
            #     [170,187,201,213,199,185],
            #     [173,202,188],
            #     [191,205],
            #     [194,208,206],
            #     [197,211,209],
            #     [200,214,212],
            #     [203,215]
            #     ]
            self._hexagons = Hexagons(sites, complete_to_eight=True) # we complete to 8 to enable sampling by hexagon

        return self._hexagons