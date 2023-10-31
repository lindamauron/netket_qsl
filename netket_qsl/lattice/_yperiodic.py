import numpy as np

from functools import partial
from jax import jit
import jax.numpy as jnp

from ._kagome import Kagome


    
class YPeriodic(Kagome):
    '''
    Kagome lattice with Y periodicity, thus the length in y-dimension must be a multiple of 4 in orer to have correct matching
    '''
    def __init__(self, a, xsize, ysize):
        '''
        a : minimal distance between two atoms
        xsize : number of triangles in the x direction
        ysize : number of triangles in the y direction
        '''
        assert ysize%4==0, 'This lattice cannot be y periodic, verify shape'
        
        
        super().__init__(a,xsize,ysize)
        
    def __repr__(self):
        '''
        Representation of the class
        '''
        return f'Kagome(N={self.N}, Y periodicity, dimension={self.xsize, self.ysize})'
    
    
    def construct_vertices(self):
        '''
        Constructs the vertices container, indicating for each vertex which atoms and triangles it possesses

        return vertices array (n_vertices,), 
        ex : for vertex #k, vertices[k]['atoms'] = [atoms it possesses], vertices[k]['triangles'] = [triangles it possesses]
        '''
        
        self.n_vertices = int(self.N/2 + self.ysize/2)
        non_border = [] # indices of the vertices which are not on the border, i.e. for which we will want the occupancy
        
        
        atoms = [[] for v in range(self.n_vertices)]
        triangles = [[] for v in range(self.n_vertices)]
        
        i = 0 #index of the vertex
        
        for y in range(self.ysize):
            for x in range(self.xsize):
                
                
                if y%2 == 0: #down triangles
                    if y==0: #first row special case
                        t1 = x + y*self.xsize
                        t2 = (self.ysize-1)*self.xsize + x # periodicity
                        
                        triangles[i].append( t1 )
                        triangles[i].append( t2 )
                        
                        atoms[i].append( 3*t1 )
                        atoms[i].append( 3*t1 + 1 )
                        atoms[i].append( 3*t2 + 1 )
                        atoms[i].append( 3*t2 + 2 )
                        
                        non_border.append( i )
                        
                        i += 1
                        
                        
                    # if the last row is down triangles, they can be treated like any normal down row:
                    else: #any row

                        t1 = x + y*self.xsize
                        t2 = t1 - self.xsize
                        
                        triangles[i].append( t1 )
                        triangles[i].append( t2 )
                        
                        atoms[i].append( 3*t1 )
                        atoms[i].append( 3*t1 + 1 )
                        atoms[i].append( 3*t2 + 1 )
                        atoms[i].append( 3*t2 + 2 )
                        
                        non_border.append( i )

                        i += 1
                        
                        
                        
                elif y%4==1 : #up triangles starting far left                
                    #here, we wont have the first row since it is down triangles by construction
                    
                    #any row

                    # left vertex
                    t2 = x + y*self.xsize

                    if x!=0:
                        t1 = t2 - self.xsize -1

                        triangles[i].append( t1 )
                        
                        atoms[i].append( 3*t1 + 1 )
                        atoms[i].append( 3*t1 + 2 )
                        
                        non_border.append( i )
                        
                        
                    triangles[i].append( t2 )
                    
                    atoms[i].append( 3*t2  )
                    atoms[i].append( 3*t2 + 1 )

                    i += 1

                    
                    # right vertex
                    t2 = x + y*self.xsize
                    t1 = t2 - self.xsize

                    triangles[i].append( t1 )
                    triangles[i].append( t2 )

                    atoms[i].append( 3*t1 )
                    atoms[i].append( 3*t1 + 2 )
                    atoms[i].append( 3*t2  )
                    atoms[i].append( 3*t2 + 2 )
                    
                    non_border.append( i )
                    
                    i += 1
                    
                    # if the line is over, we have to add the last vertex due to pbc
                    if x==self.xsize-1:
                        t = x + (y-1)*self.xsize
                        
                        triangles[i].append( t )
                        
                        atoms[i].append( 3*t + 1)
                        atoms[i].append( 3*t + 2)
                        
                        i += 1
                        
                elif y%4==3: #up triangles starting far right
                    # here, we wont have the first row since it is down triangles by construction
                    #any row
                        
                    # if we are starting the line, we have to add the lonely vertex on left due to pbc
                    if x==0:
                        t = x + (y-1)*self.xsize
                        
                        triangles[i].append( t )
                        
                        atoms[i].append( 3*t )
                        atoms[i].append( 3*t + 2)
                        
                        i += 1
                        
                        
                    # left vertex
                    t2 = x + y*self.xsize
                    t1 = t2 - self.xsize

                    triangles[i].append( t1 )
                    triangles[i].append( t2 )
                    
                    atoms[i].append( 3*t1 + 1 )
                    atoms[i].append( 3*t1 + 2 )
                    atoms[i].append( 3*t2  )
                    atoms[i].append( 3*t2 + 1 )
                    
                    non_border.append( i )
                    
                    i += 1

                    
                    # right vertex
                    t2 = x + y*self.xsize
                    if x!=self.xsize-1:
                        t1 = t2 -self.xsize + 1

                        triangles[i].append( t1 )
                        
                        atoms[i].append( 3*t1 )
                        atoms[i].append( 3*t1 + 2 )
                        
                        non_border.append( i )
                        
                    triangles[i].append( t2 )
                    
                    atoms[i].append( 3*t2  )
                    atoms[i].append( 3*t2 + 2 )


                    i += 1
        
        self.non_border = np.array(non_border)
        
        vertices = np.array([ {'atoms':[int(x) for x in atoms[k]], 'triangles':[int(v) for v in triangles[k] ]} for k in range(self.n_vertices) ])
        
        return vertices                

    
    # Computations on the lattice
    @partial(jit, static_argnums=0)
    def d(self, i, j):
        '''
        Calculates the distances between two atoms while considering the periodicity of the lattice
        Must be implemented in each subclass
        i,j : indices of the atoms between which we want the distance

        return : distance d (float)
        '''
        pi, pj = jnp.array(self.positions)[jnp.array([i,j])]

        d1 = jnp.linalg.norm( pi - pj ) #"real" distance


        # Otherwise, the distances are the ones but bringing on of the particles from the peridoice border on x, on y, or on x and y
        d2p = jnp.linalg.norm( pi - pj + self.ysize*(self.a2+self.a3) ) # y periodicity
        d2m = jnp.linalg.norm( pi - pj - self.ysize*(self.a2+self.a3) ) # y periodicity


        return jnp.min(jnp.array([d1, d2p, d2m])) 
   
  
    