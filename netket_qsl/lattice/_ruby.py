import numpy as np


from ._square import Square
from ._hexagons import Hexagons

class Ruby(Square):
    '''
    Kagome lattice in a Ruby shape as in Semeghini et al.
    Standard shape is Ruby(a=3.9, extents_down=[3,4,5,6,7,6,5], extents_up=[4,5,6,7,6,5,4])
    
    The lattice shape has to have a row of downwards triangles on the bottom and a row of upwards triangles on the top. The shape should be hexagon-like
    '''
    
    ## Initilization methods ##
    def __init__(self, a=1.0, extents_down=[3,4,5,6,7,6,5], extents_up=[4,5,6,7,6,5,4]):
        '''
        Defines the positions of each atom in the lattice
        a : unit cell size in μm
        extents_down : list of extents of the down triangles
        extents_up : list of extents of the up triangles
        Both should be compatible
        '''
        # Lattice parameters
        self.a = a
        self.a1 = a*np.array([1,0])
        self.a2 = a*np.array([1/2,np.sqrt(3)/2])
        
        self.check_shape(extents_up, extents_down) # verify if the given shape is possible
        self.extents_up = extents_up
        self.extents_down = extents_down

        self.N = int( 3*np.sum(self.extents_up)+3*np.sum(self.extents_down) )
                               
        self.construct_positions()

        self.vertices = self.construct_vertices()
        self.atoms = self.construct_atoms(self.vertices)
        self.triangles = self.construct_triangles(self.vertices, self.atoms)

        self._hexagons = None
        
        # Properties of the lattice
        self._nn = None
        self._graph = None
        self._distances = None
        self._neighbors_distances = None
        ## we compute this in any case, because the function is not jax-compatible and the lattice might be used in models
        self._n_distances = np.max(self.neighbors_distances)+1


        
    def __repr__(self):
        return f'RubyLattice(N={self.N}, shape={self.extents_up, self.extents_down})'
            
    def check_shape(self, extents_up, extents_down):
        assert len(extents_up) == len(extents_down), "The lattice should start with down triangles and end with up triangles"

        assertion = True
        flag_growing = True
        for k in range(len(extents_down)):

            if extents_down[k] < extents_up[k]:
                assertion = (extents_down[k] == extents_up[k]-1)
                flag_growing = False

            elif not flag_growing :
                assertion = (extents_down[k] == extents_up[k]+1)

            assert assertion, "extents not compatible"
            
    
    ## Constructing the connexions in the lattice ##
    def construct_vertices(self):
        '''
        Constructs the triangles container, indicating for each triangle which atoms and triangles it possesses
        
        return vertices array (n_vertices,), 
        ex : for vertex #k, vertices[k]['atoms'] = [atoms it possesses], vertices[k]['triangles'] = [triangles it possesses]
        '''
            
        self.n_vertices = int( self.N/2 + 0.5*(self.extents_down[0] + self.extents_up[-1] + 2*len(self.extents_down) ) )

        # positions container
        #vertices = np.array([ {'atoms':[], 'triangles':vertex[k]} for k in range(n_vertices) ])

        vertex = [[] for j in range(self.n_vertices)]
        atoms = [[] for j in range(self.n_vertices)]
        non_border = []

        index_vertex = 0

        for k in range(len(self.extents_down)): # 7 lines of down-up triangles

            # growing part
            if self.extents_down[k] < self.extents_up[k]:
                '''
                np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i # Down triangle i of line k
                np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i # Up triangle i of line k 
                '''

                for i in range(self.extents_down[k]): ## vertical vertices 

                    if k == 0:

                        vertex[index_vertex].append( i ) # first down triangle
                        atoms[index_vertex].append( 3*i )
                        atoms[index_vertex].append( 3*i + 1 )
                        index_vertex += 1

                    else : 

                        vertex[index_vertex].append( np.sum(self.extents_up[:k-1]) + np.sum(self.extents_down[:k]) + i ) ## up triangle on line k-1
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k-1]) + np.sum(self.extents_down[:k]) + i) + 1 )
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k-1]) + np.sum(self.extents_down[:k]) + i) + 2 )
                        
                        vertex[index_vertex].append( np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i ) ## down triangle on line k
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i) )
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i) + 1 )
                        
                        non_border.append( index_vertex )
                        
                        index_vertex += 1

                for i in range(self.extents_up[k]): ## pi/6 vertices, i.e. the ones on the common line (treated by pairs)

                    if i == 0: # first element only has one triangle
                        vertex[index_vertex].append( np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) ) #first up triangle only
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1])) )
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1])) + 1 )
                        
                        index_vertex += 1

                        
                        vertex[index_vertex].append( np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) ) #first up triangle
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1])) )
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1])) + 2 )
                        
                        
                        vertex[index_vertex].append( np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) ) #first down triangle
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k])) )
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k])) + 2 )
                        
                        non_border.append( index_vertex )
                                          
                        index_vertex += 1


                    elif i == self.extents_up[k]-1: # last element only has one triangle
                        vertex[index_vertex].append( np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i-1 ) # last down triangle on the left
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i-1) + 1 )
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i-1) + 2 )
                        
                        vertex[index_vertex].append( np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i ) # last up triangle on the right
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i) )
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i) + 1 )
                        
                        non_border.append( index_vertex )

                        index_vertex += 1

                        
                        vertex[index_vertex].append( np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i ) # last up triangle on the right
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i) )
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i) + 2 )
                        
                        index_vertex += 1

                    else : 

                        # first vertex : down triangle on the left & up triangle on the right
                        vertex[index_vertex].append( np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i-1 ) #line k, place i-1
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i-1) + 1 )
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i-1) + 2 )
                        
                        vertex[index_vertex].append( np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i ) #line k+1, place i
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i) )
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i) + 1 )
                        
                        non_border.append( index_vertex )

                        index_vertex += 1

                        # second vertex : up triangle on left, down triangle on right
                        vertex[index_vertex].append( np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i ) #line k, place i
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i) )
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i) + 2 )
                        
                        vertex[index_vertex].append( np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i ) #line k, place i
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i) )
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i) + 2 )
                        
                        non_border.append( index_vertex )

                        index_vertex += 1

            # shrinking part
            else:
                '''
                np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i # Down triangle i of line k
                np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i # Up triangle i of line k 
                '''
                # both-at-same-size part, i.e. changing layer
                if self.extents_up[k] == self.extents_down[k-1] and self.extents_down[k] == self.extents_up[k-1]:
                    for i in range(self.extents_down[k]): # vertical vertices 
                        vertex[index_vertex].append( np.sum(self.extents_up[:k-1]) + np.sum(self.extents_down[:k]) + i ) ## up triangle on line k-1
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k-1]) + np.sum(self.extents_down[:k]) + i) + 1 )
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k-1]) + np.sum(self.extents_down[:k]) + i) + 2 )
                        
                        vertex[index_vertex].append( np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i ) ## down triangle on line k
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i) )
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i) + 1 )
                        
                        non_border.append( index_vertex )

                        index_vertex += 1


                for i in range(self.extents_down[k]): ## pi/6 vertices, i.e. the ones on the common line

                    if i == 0: # first element only has one triangle
                        vertex[index_vertex].append( np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i ) #first down triangle on the right
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i) )
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i) + 2 )
                        
                        index_vertex += 1

                        
                        vertex[index_vertex].append( np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i ) #first down triangle
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i) + 1 )
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i) + 2 )
                        
                        vertex[index_vertex].append( np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i ) #first up triangle
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i) )
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i) + 1 )
                        
                        non_border.append( index_vertex )

                        index_vertex += 1

                    elif i == self.extents_down[k]-1: # last element only has one triangle
                        # one before end
                        vertex[index_vertex].append( np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i-1 ) #line k, place i-1
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i-1) )
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i-1) + 2 )
                        
                        vertex[index_vertex].append( np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i ) #line k, place i
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i) )
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i) + 2 )
                        
                        non_border.append( index_vertex )

                        index_vertex += 1

                        
                        vertex[index_vertex].append( np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i ) #last down triangle
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i) + 1 )
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i) + 2 )
                        
                        index_vertex += 1

                    else : 

                        # left vertex : up triangle on the left, down triangle on the right
                        vertex[index_vertex].append( np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i-1 ) #line k, place i-1
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i-1) )
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i-1) + 2 )
                        
                        vertex[index_vertex].append( np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i ) #line k, place i
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i) )
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i) + 2 )
                        
                        non_border.append( index_vertex )

                        index_vertex += 1

                        
                        # right vertex : down triangle on the left, up triangle on the right
                        vertex[index_vertex].append( np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i ) #line k, place i
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i) + 1 )
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k]) + i) + 2 )
                        
                        vertex[index_vertex].append( np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i ) #line k, place i
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i) )
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i) + 1 )
                        
                        non_border.append( index_vertex )

                        index_vertex += 1
                
                for i in range(self.extents_up[k]): # vertical vertices 
                    if k == len(self.extents_up)-1: # last up triangles : each vertex only has one 
                        vertex[index_vertex].append( np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i ) ## up triangle on line k-1
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i) + 1 )
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i) + 2 )
                        
                        index_vertex += 1
                    else:
                        vertex[index_vertex].append( np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i ) ## up triangle on line k-1
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i) + 1 )
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k]) + np.sum(self.extents_down[:k+1]) + i) + 2 )
                        
                        vertex[index_vertex].append( np.sum(self.extents_up[:k+1]) + np.sum(self.extents_down[:k+1]) + i ) ## down triangle on line k
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k+1]) + np.sum(self.extents_down[:k+1]) + i) )
                        atoms[index_vertex].append( 3*(np.sum(self.extents_up[:k+1]) + np.sum(self.extents_down[:k+1]) + i) + 1 )
                        
                        non_border.append( index_vertex )

                        index_vertex += 1


        vertices = np.array([ {'atoms':[int(x) for x in atoms[k]], 'triangles':[int(v) for v in vertex[k] ]} for k in range(self.n_vertices) ])
        self.non_border = np.array(non_border)
        return vertices


    def construct_positions(self):
        '''
        constructs the positions of the atoms on the latticce
        '''
                               
        # positions container
        self.positions = []

        i = 0
        # Start with the down triangles
        for k,line in enumerate(self.extents_down): #0,1,2,3,4,5,6 & 3,4,5,6,7,6,5
            for el in range(line): #0,1,2,3 ; 0,1,2,3,4 ; ...

                if i==0 : # Very first triangle
                    pos0 = np.array([0,0])
                    pos1 = pos0 + self.a1
                    pos2 = pos0 + self.a2


                else:
                    if el == 0 : #first element of line
                        if self.extents_down[k] > self.extents_down[k-1]: #first triangle is on the left
                            pos0 = self.positions[i-3*self.extents_down[k-1]-3*self.extents_up[k-1]] + 4*(self.a2 - self.a1)                
                        else: # first triangle is on the right
                            pos0 = self.positions[i-3*self.extents_down[k-1]-3*self.extents_up[k-1]] + 4*self.a2
                            
                        # Positions given the first element of the triangle
                        pos1 = pos0 + self.a1
                        pos2 = pos0 + self.a2


                    else : # just translate the first triangle of the line
                        pos0 = self.positions[i-3] + 4*self.a1
                        pos1 = self.positions[i-2] + 4*self.a1
                        pos2 = self.positions[i-1] + 4*self.a1

                # Save everything
                self.positions.append( pos0 )
                self.positions.append( pos1 )
                self.positions.append( pos2 )

                # keep track of the indices
                i += 3

            # Now do the down triangles
            for el in range(self.extents_up[k]):
                if el == 0 : #first element of line
                    if self.extents_up[k] > self.extents_down[k]: #first triangle is on the left of down triangle
                        pos0 = self.positions[i-3*self.extents_down[k]] -2*self.a1+self.a2               
                    else: # first triangle is on the right
                        pos0 = self.positions[i-3*self.extents_down[k]] +2*self.a1+self.a2

                    pos1 = pos0 + self.a2-self.a1
                    pos2 = pos0 + self.a2


                else : 
                    pos0 = self.positions[i-3] + 4*self.a1
                    pos1 = self.positions[i-2] + 4*self.a1
                    pos2 = self.positions[i-1] + 4*self.a1


                self.positions.append( pos0 )
                self.positions.append( pos1 )
                self.positions.append( pos2 )

                i += 3
        self.positions = np.array(self.positions)

    
    
    @property
    def hexagons(self):
        '''
        Defines the (partial) hexagons of the lattice
        '''
        if not self._hexagons:
            if self.extents_down==[3,4,5,6,5] and self.extents_up==[4,5,6,5,4]:
                sites = np.array([
                    [26,40,54,69,52,38],
                    [29,43,57,72,55,41],
                    [53,70,87,102,85,68],
                    [56,73,90,105,88,71],
                    [59,76,93,108,91,74]
                ])
                self._hexagons = Hexagons(sites)

            elif self.extents_down==[3,4,5,6,7,6,5] and self.extents_up==[4,5,6,7,6,5,4]:
                sites = np.array([
                        [0,9],
                        [3,12,1],
                        [6,15,4],
                        [18,7],
                        [10,21,33],
                        [2,13,24,36,22,11],
                        [5,16,27,39,25,14],
                        [8,19,30,42,28,17],
                        [45,31,20],
                        [34,48,63],
                        [23,37,51,66,49,35],
                        [26,40,54,69,52,38],
                        [29,43,57,72,55,41],
                        [32,46,60,75,58,44],
                        [78,61,47],
                        [64,81,99],
                        [50,67,84,102,82,65],
                        [53,70,87,105,85,68],
                        [56,73,90,108,88,71],
                        [59,76,93,111,91,74],
                        [62,79,96,114,94,77],
                        [117,97,80],
                        [100,120],
                        [83,103,123,141,121,101],
                        [86,106,126,144,124,104],
                        [89,109,129,147,127,107],
                        [92,112,132,150,130,110],
                        [95,115,135,153,133,113],
                        [98,118,138,156,136,116],
                        [139,119],
                        [122,142,159],
                        [125,145,162,177,160,143],
                        [128,148,165,180,163,146],
                        [131,151,168,183,166,149],
                        [134,154,171,186,169,152],
                        [137,157,174,189,172,155],
                        [140,175,158],
                        [161,178,192],
                        [164,181,195,207,193,179],
                        [167,184,198,210,196,182],
                        [170,187,201,213,199,185],
                        [173,190,204,216,202,188],
                        [176,205,191],
                        [194,208],
                        [197,211,209],
                        [200,214,212],
                        [203,217,215],
                        [206,218]
                    ], dtype=object)

                self._hexagons = Hexagons(sites)

            elif self.extents_down==[4,5,6,7,8,7,6,5] and self.extents_up==[5,6,7,8,7,6,5,4]:
                sites = np.array([
                    [0,12],
                    [3,15,1],
                    [6,18,4],
                    [9,21,7],
                    [24,10],
                    [13,27,42],
                    [2,16,30,45,28,14],
                    [5,19,33,48,31,17],
                    [8,22,36,51,34,20],
                    [11,25,39,54,37,23],
                    [57,40,26],
                    [43,60,78],
                    [29,46,63,81,61,44],
                    [32,49,66,84,64,47],
                    [35,52,69,87,67,50],
                    [38,55,72,90,70,53],
                    [41,58,75,93,73,56],
                    [96,76,59],
                    [79,99,120],
                    [62,82,102,123,100,80],
                    [65,85,105,126,103,83],
                    [68,88,108,129,106,86], #hole
                    [71,91,111,132,109,89],
                    [74,94,114,135,112,92],
                    [77,97,117,138,115,95],
                    [141,118,98],
                    [121,144],
                    [101,124,147,168,145,122],
                    [104,127,150,171,148,125],
                    [107,130,153,174,151,128],
                    [110,133,156,177,154,131],
                    [113,136,159,180,157,134],
                    [116,139,162,183,160,137],
                    [119,142,165,186,163,140],
                    [166,143],
                    [146,169,189],
                    [149,172,192,210,190,170],
                    [152,175,195,213,193,173],
                    [155,178,198,216,196,176],
                    [158,181,201,219,199,179],
                    [161,184,204,222,202,182],
                    [164,187,207,225,205,185],
                    [208,188,167],
                    [191,211,228],
                    [194,214,231,246,229,212],
                    [197,217,234,249,232,215],
                    [200,220,237,252,235,218],
                    [203,223,240,255,238,221],
                    [206,226,243,258,241,224],
                    [209,244,227],
                    [230,247,261],
                    [233,250,264,276,262,248],
                    [236,253,267,279,265,251],
                    [239,256,270,282,268,254],
                    [242,259,273,285,271,257],
                    [245,274,260],
                    [263,277],
                    [266,280,278],
                    [269,283,281],
                    [272,286,284],
                    [275,287]
                ], dtype=object)
                self._hexagons = Hexagons(sites)

        return self._hexagons

    def plot(self, ax=None, to_draw=None, annotate=False, plot_lines=False):
        '''
        Plots the lattice on a chosen figure with annotation of the atoms
        For example : 

        fig, ax = plt.subplots(1,1, figsize=(6,4))
        lattice.plot(ax, True, True,-np.ones(N))
        ax.axis('off')
        plt.show() 

        ax : AxesSubPlot on which to plot the lattice
        annotate : boolean indicating wether we write down the indices of the atoms
        plot_lines : boolean indicating wether we want the lattice structure (lines) plotted as well
                     (helps for visibility but takes quite longer to load, i.e. do not do many times)
        to_draw : what to draw on the lattice
                    can be: a state (ndarray, |g> in black, |r> in red)
                            a topological operator (sites on the contour in blue, rest in black)
                            a product op topological operators (each operator has its colour following the default cycle)        
        '''
        ax = super().plot(ax, annotate=annotate, plot_lines=False, to_draw=to_draw)
            
        # If indicated, plot the lattice's structure    
        if plot_lines:
            s = np.sqrt(3)
            xsize = np.max(self.extents_down)
            ysize = len(self.extents_down)

            pos = self.positions/self.a

            # do r0 + [-4/2, -4s/2] to start from lower
            # do r1 + [4/2, 4s/2] to get further
            x = np.array([-27/2, 9/2])
            y = np.array([-5*s/2, 31*s/2])
            for k in range(-4,xsize+2):
                ax.plot( x+4*k, y, color='k', linestyle=':')

            # to the right
            # do r0 + [-4/2, 4s/2] to get higher
            # do r1 + [4/2, -4s/2] to get lower
            x = np.array([-31/2, 5/2])
            y = np.array([31*s/2, -5*s/2])
            for k in range(-2,xsize+2):
                ax.plot( x+4*k, y, color='k', linestyle=':')

            # horizontal
            # do r0 + [-4/2, 0] to start from lefter
            # do r1 + [4/2, 0] to get righter
            x = np.array([-31/2, 17/2+ 4*xsize])
            y = np.array([5*s/2, 5*s/2])
            for k in range(-2,ysize//2+5):
                ax.plot( x, y+4*k*s/2, color='k', linestyle=':')

                
            ax.set_xlim([np.min(pos[:,0])-2,np.max(pos[:,0])+2])
            ax.set_ylim([np.min(pos[:,1])-2,np.max(pos[:,1])+2])


        return
    

