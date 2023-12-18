import numpy as np
import jax.numpy as jnp
from ._utils import neighbors

class Hexagons:
    def __init__(self, sites):
        self.sites = sites.copy()
        
        filled_hex = []
        nonbord_hex = []
        
        # we fill the hexagons either with an even number of the same site (Pi^2 = 1) or a path on one triangle (P0*P1*P2=1)
        # this is not very clean, but works methematcally and allows to jax the container
        for i,h in enumerate(sites):
            if len(h) == 6:
                filled_hex.append( h.copy() )
                nonbord_hex.append( h.copy() )
            elif len(h) == 5:
                raise ValueError(f'This hexagon {h} has length 5 which should be impossible')
            elif len(h) == 4:
                new_hex = h.copy()
                
                new_hex.append( 0 )
                new_hex.append( 0 )

                filled_hex.append( new_hex )
                
            elif len(h) == 3:
                new_hex = h.copy()
                
                new_hex.append( 0 )
                new_hex.append( 1 )
                new_hex.append( 2 )

                filled_hex.append( new_hex )
            elif len(h) == 2:
                new_hex = h.copy()
                
                new_hex.append( 0 )
                new_hex.append( 0 )
                new_hex.append( 0 )
                new_hex.append( 0 )

                filled_hex.append( new_hex )
                
            elif len(h) == 1:
                new_hex = h.copy()
                
                new_hex.append( 0 )
                new_hex.append( 1 )
                new_hex.append( 2 )
                
                new_hex.append( 0 )
                new_hex.append( 0 )

                filled_hex.append( new_hex )


        self.filled = jnp.array(filled_hex)
        self.nonbord = jnp.array(nonbord_hex)
        
        bulk = []
        for h in self.nonbord:
            # all sites around the hexagon
            triangles = np.array(neighbors(np.array(h))).reshape(-1)


            if np.isin(triangles, self.nonbord.reshape(-1)).all():
                bulk.append( h.copy() )
                
        self.bulk = jnp.array(bulk)
                
                
        """
        self.distances = jnp.array([
            [0,1,1,1],
            [1,0,1,1],
            [1,1,0,1],
            [1,1,1,0]
        ])
        
        self.positions = a*jnp.array([[ 0.5       ,  2.59807621],
                                   [ 4.5       ,  2.59807621],
                                   [-1.5       ,  6.06217783],
                                   [ 2.5       ,  6.06217783]
                                  ])
        
        self.relative_pos = jnp.array([2, 4, 0, 2, 4, 0, 3, 1, 5, 3, 1, 5, 2, 4, 0, 2, 4, 0, 3, 1, 5, 3, 1, 5])
        self.hex_pos = jnp.take_along_axis(jnp.arange(24)//6, jnp.argsort(self.atoms.reshape(-1)), axis=0)
        self.R = self.distances[self.hex_pos][...,self.hex_pos]
        """