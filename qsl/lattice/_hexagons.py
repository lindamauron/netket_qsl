import numpy as np
import jax.numpy as jnp
from ._utils import neighbors

class Hexagons:
    def __init__(self, sites, complete_to_eight=False):
        '''
        complete to eight: flag indicating whether we complete the hexagons to 8 sites (instead of 6by default)
                            this allows to include the hexagons of 5 sites
        '''
        self._sites = sites.copy()
        
        filled_hex = []
        nonbord_hex = []
        
        # we fill the hexagons either with an even number of the same site (Pi^2 = 1) or a path on one triangle (P0*P1*P2=1)
        # this is not very clean, but works mathematcally and allows to jax the container
        if not complete_to_eight:
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
        else:
            for i,h in enumerate(sites):
                if len(h) == 6:
                    new_hex = h.copy()
                    nonbord_hex.append( new_hex )

                    new_hex.append( 0 )
                    new_hex.append( 0 )
                    filled_hex.append( new_hex )
                elif len(h) == 5:
                    new_hex = h.copy()
                    
                    new_hex.append( 0 )
                    new_hex.append( 1 )
                    new_hex.append( 2 )

                    filled_hex.append( new_hex )
                elif len(h) == 4:
                    new_hex = h.copy()
                    
                    new_hex.append( 0 )
                    new_hex.append( 0 )
                    new_hex.append( 0 )
                    new_hex.append( 0 )

                    filled_hex.append( new_hex )
                    
                elif len(h) == 3:
                    new_hex = h.copy()
                    
                    new_hex.append( 0 )
                    new_hex.append( 1 )
                    new_hex.append( 2 )
                    new_hex.append( 0 )
                    new_hex.append( 0 )

                    filled_hex.append( new_hex )
                elif len(h) == 2:
                    new_hex = h.copy()
                    
                    new_hex.append( 0 )
                    new_hex.append( 0 )
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
                    new_hex.append( 0 )
                    new_hex.append( 0 )

                    filled_hex.append( new_hex )

        self._filled = jnp.array(filled_hex)
        self._nonbord = jnp.array(nonbord_hex)[:,:6]
        
        bulk = []
        for h in self.nonbord:
            # all sites around the hexagon
            triangles = np.array(neighbors(np.array(h))).reshape(-1)

            if np.isin(triangles, self.nonbord.reshape(-1)).all():
                bulk.append( h.copy() )
                
        self._bulk = jnp.array(bulk)

    def __getitem__(self, item):
        return self.sites[item]
    
    @property
    def sites(self):
        return self._sites
    
    @property
    def bulk(self):
        return self._bulk
    
    @property
    def filled(self):
        return self._filled
    
    @property
    def nonbord(self):
        return self._nonbord