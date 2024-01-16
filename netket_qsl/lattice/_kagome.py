import numpy as np
import networkx as nx
import netket as nk
import abc
import matplotlib.pyplot as plt
import warnings 

from functools import partial
from jax import vmap, jit
import jax.numpy as jnp
from netket.utils.types import Array

from ._hexagons import Hexagons
    
class Kagome:
    r'''
    Superclass defining the rectangular kagome lattice
    There 4 sublcalsses for various boundary conditions : 
    - Torus for X and Y periodicity
    - YPeriodic for Y periodicity
    - XPeriodic for X periodicity
    - Square for no periodicity
    
    Once created, the instance allows to know the connexions between the atoms (also triangles and vertices),
    to get the graph, netket graph and plot of the lattice,
    as well as the operators (potential for the hamiltonian) corresponding (i.e. evaluating distances)

    Available properties : 
    - lattice vectors
    - N, n_vertices, n_triangles
    - hexagons, vertices, triangles, atoms, non_border
    - positions, d, distances
    - nn, find_neighbors
    - graph, nx_graph
    - plot_lattice, plot_connexions
    - dimer_probs
    '''

    def __init__(self, a, xsize, ysize):
        '''
        a : minimual distance between two atoms (i.e. length of the side of a triangle)
        xsize : number of triangles in the x direction
        ysize : number of triangles in the y direction
        '''

        # Lattice cells
        self.a = a
        self.a1 = a*np.array([1,0])
        self.a2 = a*np.array([1/2,np.sqrt(3)/2])
        self.a3 = a*np.array([-1/2,np.sqrt(3)/2]) # is not a basis vector but can be very useful
            
        # save the geometry
        self.N = int(3*xsize*ysize)
        self.ysize = int(ysize)
        self.xsize = int(xsize)

        # Constructing the connexions in the lattice
        self.vertices = self.construct_vertices()
        self.atoms = self.construct_atoms(self.vertices)
        self.triangles = self.construct_triangles(self.vertices, self.atoms)
        
        self.positions = self.construct_pos()

        self._hexagons = None
        
        # Properties of the lattice
        self._nn = None
        self._graph = None
        self._distances = None
        self._neighbors_distances = None
        self._n_distances = None
        
        
    # Construction methods (are called at init in any case)
    def construct_pos(self):
        '''
        constructs the positions of the atoms on the lattice
        
        return : container with all atoms positions (N,2)
        '''
        
        positions = []
        
        i = 0 #index of the atom
        for y in range(self.ysize):
            for x in range(self.xsize):
                if i==0:
                    pos0 = np.array([0,0])
                    pos1 = pos0 + self.a1
                    pos2 = pos0 + self.a2
                    
                    pos0_prev_line = pos0
                    
                elif x==0: #first triangle on the line
                    
                    if y%4==0: #right down triangles
                        pos0 = pos0_prev_line + self.a1 + 3*self.a3
                        pos1 = pos0 + self.a1
                        pos2 = pos0 + self.a2
                        
                        pos0_prev_line = pos0
                        
                    if y%4==1: #left up triangles
                        pos0 = pos0_prev_line + self.a3 - self.a1
                        pos1 = pos0 + self.a3
                        pos2 = pos0 + self.a2
                        
                        pos0_prev_line = pos0
                        
                    if y%4==2: #left down triangles
                        pos0 = pos0_prev_line + self.a1 +3*self.a3
                        pos1 = pos0 + self.a1
                        pos2 = pos0 + self.a2
                        
                        pos0_prev_line = pos0                        
                        
                    if y%4==3: # right up triangles
                        pos0 = pos0_prev_line + self.a2 + 2*self.a1
                        pos1 = pos0 + self.a3
                        pos2 = pos0 + self.a2
                        
                        pos0_prev_line = pos0
                        
                else : # any triangle on the line
                    pos0 = positions[i-3] + 4*self.a1
                    pos1 = positions[i-2] + 4*self.a1
                    pos2 = positions[i-1] + 4*self.a1
                    
                positions.append( pos0 )
                positions.append( pos1 )
                positions.append( pos2 )

                i += 3
        return np.array(positions)
    
    @abc.abstractmethod
    def construct_vertices(self):
        '''
        Constructs the vertices container, indicating for each vertex which atoms and triangles it possesses
        Must be implemented in each subclass as it depends on the pbc

        return vertices array (n_vertices,), 
        ex : for vertex #k, vertices[k]['atoms'] = [atoms it possesses], vertices[k]['triangles'] = [triangles it possesses]
        '''
        
        
    def construct_atoms(self, vertices=None):
        '''
        Constructs the atoms container, indicating for each atom which triangle and vertices it belongs to
        vertices : container of the vertices of the lattice
        
        return atoms array (N,), 
        ex : for atom #k, atoms[k]['triangles'] = [triangle it belongs to], atoms[k]['vertices'] = [vertices it belongs to]
        '''
        atoms = np.array([ {'triangles':int(k/3), 'vertices':[]} for k in range(self.N) ])

        if vertices is None:
            vertices = self.construct_vertices()
            
            
        for i in range(self.N):
            
            # find all vertices where atom i appears
            for k in range(self.n_vertices):
                if np.isin(self.vertices[k]['atoms'], i).any():
                    atoms[i]['vertices'].append(k)


        return atoms
    
    
    def construct_triangles(self, vertices=None, atoms=None):
        '''
        Constructs the triangles container, indicating for each triangle which atoms and vertices it possesses
        vertices : container of the vertices of the lattice
        atoms : container of the atoms of the lattice
        
        return triangles array (n_triangles,), 
        ex : for triangle #k, triangles[k]['atoms'] = [atoms it possesses], triangles[k]['vertices'] = [vertices it possesses]
        '''
        
        self.n_triangles = int(self.N/3)
        
        triangles = np.array([ {'atoms':[], 'vertices':[]} for k in range(self.n_triangles) ])


        if vertices is None:
            vertices = self.construct_vertices()
        if atoms is None:
            atoms = self.construct_atoms(vertices)
        
        for i in range(self.n_triangles):
            # the indices of the atoms in this triangle
            i0 = int(3*i)
            i1 = int(3*i+1)
            i2 = int(3*i+2)

            # update triangle's atoms
            triangles[i]['atoms'] = [i0, i1, i2]

            # each vertex is the common one between atoms in the triangle
            # vertex[j] = vertex opposite to atom j in triangle
            triangles[i]['vertices'].append( np.intersect1d(atoms[ i1 ]['vertices'], atoms[ i2 ]['vertices'], assume_unique=True)[0] )
            triangles[i]['vertices'].append( np.intersect1d(atoms[ i0 ]['vertices'], atoms[ i2 ]['vertices'], assume_unique=True)[0] )
            triangles[i]['vertices'].append( np.intersect1d(atoms[ i0 ]['vertices'], atoms[ i1 ]['vertices'], assume_unique=True)[0] )
    
    
        return triangles

    @property
    def hexagons(self):
        '''
        Generates the hexagon container that indicates which atoms are contained in each hexagon s.t. the atoms in hexagon number k are given by hexagons()[k]
        The hexagon numerotation starts from bottom left
        This implementation actually desccribes the hexagons on a torus, but is available for any lattice boudary condition
        
        return : container with the indices of the atoms contained in each hexagon (ndarray (N/6, 6) )
        '''
        if self._hexagons is None :
            def plaquette(idx,idy):
                '''
                Allows to find the index of the triangle with position (idx,idy) on the lattice (2d indices)
                (idx, idy) : two-dimensional indices of the said plaquette
                
                return : 1d index (int) telling the number of the concerned triangle
                '''
                x = idx%self.xsize
                y = idy%self.ysize
                
                return self.xsize*y + x


            # container
            sites = []

            for y in range(0,self.ysize,2):
                for x in range(self.xsize):
                    
                    # indices of the triangles we will consider for the hexagon
                    # always from bottom triangle in anti-clockwise way
                    # be careful since all rows are not the same 
                    if y%4==0:
                        idx = [plaquette(x,y), plaquette(x+1,y+1), plaquette(x+1,y+2), plaquette(x,y+3), plaquette(x,y+2), plaquette(x,y+1)]
                    else:
                        idx = [plaquette(x,y), plaquette(x,y+1), plaquette(x,y+2), plaquette(x,y+3), plaquette(x-1,y+2), plaquette(x-1,y+1)]
                        
                    # All the atoms in the concerned triangles
                    triangles = [self.triangles[i]['atoms'] for i in idx]

                    # in each triangle, we select the atom we are looking for
                    atoms = [ triangles[0][2], triangles[1][1], triangles[2][0], triangles[3][0], triangles[4][1], triangles[5][2] ]
                    
                    # store result
                    sites.append( atoms )

            self._hexagons = Hexagons(sites)

        return self._hexagons


    # Representation of the lattice      
    def nx_graph(self):
        '''
        Constructs the Networkx graph of the lattice
        Each node contains the following info : 
            'pos':position of the atom in the lattice 
            'vertex':vertex to which it is connected 
            'triangle':triangle in which it is present
        The edges correspond to the triangles connexions
            
        returns : the networkx graphic of the lattice
        '''
        G = nx.Graph()
        for i in range(self.N):
            G.add_node(i, pos=self.positions[i], vertex=self.atoms[i]['vertices'], triangle=self.atoms[i]['triangles'])

        for v in self.vertices:
            if len(v['atoms']) == 2:
                G.add_edges_from( nx.complete_graph( v['atoms']).edges )
            else:
                G.add_edges_from( nx.cycle_graph( np.array(v['atoms'])[np.array([0,1,3,2], dtype=int)]).edges )
            
        return G

        
    @property
    def graph(self):
        '''
        Generates the netket graph of the lattice
        '''
        if self._graph is None:
            G = self.nx_graph()

            self._graph = nk.graph.Graph(list(G.edges))
            self._graph.from_networkx(G)
        
        return self._graph


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
        if not ax:
            fig, ax = plt.subplots(1,1, figsize=(6,4))

        # Plot operators, partitions, etc
        if hasattr(to_draw, "draw") and callable(to_draw.draw):
            state, colors = to_draw.draw()

        # if hasattr(to_draw, "sites"):
        #     state = np.zeros(self.N, dtype=int)
        #     state[to_draw.sites] = 1
            
        #     colors = ['k', 'b']

        # # Plot product of operators
        # elif hasattr(to_draw, "operators"):
            # for op in to_draw.operators:
            #     if not hasattr(op, "sites"):
            #         raise AttributeError(f'The input to draw has operators but {op} has no attribute sites')
                
            # assert len(to_draw.operators) <= 10, 'The drawing for so many operators is not defined'

            # colors = ['k']
            # state = np.zeros(self.N, dtype=int)
            # for k,op in enumerate(to_draw.operators):
            #     state[op.sites] = k+1
            #     colors.append(f'C{k}')
            
        # plot samples, if many, plot only the first one
        elif isinstance(to_draw, Array) and to_draw.shape[-1] == self.N:
            if to_draw.ndim != 1:
                warnings.warn(f'Array to plot has shape {to_draw.shape}, only first configuration is shown')
                to_draw = to_draw.reshape(-1,self.N)[0]
            
            state = np.array( (1+to_draw)/2, dtype=int)
            colors = ['k', 'r']

        elif isinstance(to_draw, Array) or isinstance(to_draw, list):
            to_draw = np.array(to_draw).reshape(-1)
            if not (to_draw<self.N).all() or not (to_draw>=0).all():
                raise ValueError(f'A list of sites must have indices in the right range, instead got range [{to_draw.min()}, to_draw.max()]')
            
            state = np.zeros(self.N, dtype=int)
            state[to_draw] = 1

            colors = ['k', 'r']
        else:
            state = np.zeros(self.N, dtype=int)
            colors = ['k']
        c = np.tile(colors, (self.N,))

        # c = np.tile(['darkgreen', 'r'], (self.N,))
        # # by default, we just do all green i.e. |g>
        # if sample is None:
        #     sample = -np.ones(self.N)
        # assert sample.ndim == 1, 'The sample to draw must be one dimensional'

        # state = np.array( (1+sample)/2, int)

        pos = self.positions/self.a
        
        # represent each atom on the right position with the right color
        #for i in range(self.N):
        ax.scatter( pos[:,0], pos[:,1], color=c[state], zorder=10 )   

        # annotate if indicated
        if annotate :
            for i in range(self.N):
                ax.annotate(i,xy=pos[i], xytext=(5, 2), textcoords='offset points',ha='right',va='bottom')            

        # If indicated, plot the lattice's structure    
        if plot_lines:
            s = np.sqrt(3)
            
            # special often-used case where I hard-coded the structure
            if self.N == 24:

                ax.plot( [-5/2, 1/2], [s/2, 7*s/2], color='k', linestyle='-')
                ax.plot( [1/2, 9/2], [-s/2, 7*s/2], color='k', linestyle='-')
                ax.plot( [9/2, 11/2], [-s/2, s/2], color='k', linestyle='-')

                ax.plot( [1/2, -5/2], [-s/2, 5*s/2], color='k', linestyle='-')
                ax.plot( [9/2, 1/2], [-s/2, 7*s/2], color='k', linestyle='-')
                ax.plot( [11/2, 9/2], [5*s/2, 7*s/2], color='k', linestyle='-')

                ax.plot( [-5/2, 11/2], [s/2, s/2], color='k', linestyle='-')
                ax.plot( [-5/2, 11/2], [5*s/2, 5*s/2], color='k', linestyle='-')
            
            # grid part (works on all lattices)

            # to the left
            # do r0 + [-4/2, -4s/2] to start from lower
            # do r1 + [4/2, 4s/2] to get further
            x = np.array([-27/2, 13/2])
            y = np.array([-13*s/2, 27*s/2])
            for k in range(-2,self.xsize+3):
                ax.plot( x+4*k, y, color='k', linestyle=':')

            # to the right
            # do r0 + [-4/2, 4s/2] to get higher
            # do r1 + [4/2, -4s/2] to get lower
            x = np.array([-35/2, 5/2])
            y = np.array([27*s/2, -13*s/2])
            for k in range(-2,self.xsize+4):
                ax.plot( x+4*k, y, color='k', linestyle=':')

            # horizontal
            # do r0 + [-4/2, 0] to start from lefter
            # do r1 + [4/2, 0] to get righter
            x = np.array([-27/2, 17/2+ 4*self.xsize])
            y = np.array([-3*s/2, -3*s/2])
            for k in range(-2,self.ysize//2+3):
                ax.plot( x, y+4*k*s/2, color='k', linestyle=':')

            ax.set_xlim([-5,pos[-1,0]+3])
            ax.set_ylim([-3,pos[-1,1]+3])

        ax.axis('off')
        return ax
        
    def draw(self, ax, annotate=False, plot_links=False):
        '''
        Plots the lattice on a chosen figure with annotation of the atoms
        ax : AxesSubPlot on which to plot the lattice
        annotate : boolean indicating wether we write down the indices of the atoms
        plot_links : boolean indicating wether we want the lattice structure (lines) plotted as well
        '''
        G = self.nx_graph()

        pos=nx.get_node_attributes(G,'pos')

        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=30, node_color='darkgreen')
        if annotate : nx.draw_networkx_labels(G, pos, ax=ax, horizontalalignment='left', verticalalignment='top', font_size=8, font_color='darkgreen')
        if plot_links : nx.draw_networkx_edges(G, ax=ax, pos=pos)
        
        return

    @property
    def distances(self):
        '''
        Distance tensor between the atoms (taking periodiciy into account)
        
        return : ndarray (N,N) containing the physical distances d[i,j] = |r_i - r_j|
        '''
        if self._distances is None:
            # we vmap it on i and j to have all (i,j) combinations
            self._distances = np.array( vmap( vmap(self.d, in_axes=(None,0)), in_axes=(0,None) )(np.arange(self.N), np.arange(self.N)) )

        return self._distances
        
    @abc.abstractmethod
    def d(self, i, j):
        '''
        Calculates the distances between two atoms while considering the periodicity of the lattice
        Must be implemented in each subclass
        i,j : indices of the atoms between which we want the distance
        
        return : distance d = |r_i - r_j| (float)
        '''
        
        
    @property
    def nn_matrix(self):
        '''
        Computes the tensor of 1st nearest neighbours of the lattice, i.e. the atoms on the same triangle in the lattice
        
        returns : array (N,2) where nn[k] = [2 nearest neighbors of atom k]
        '''
        if self._nn is None:
            self._nn = np.zeros( (self.N, 2), dtype=int )
            
            # construct the graph with the edges defined by the vertices/triangles
            G = nx.Graph()
            for i in range(self.N):
                G.add_node(i, pos=self.positions[i], vertex=self.atoms[i]['vertices'], triangle=self.atoms[i]['triangles'])

            for t in self.triangles:
                G.add_edges_from( nx.complete_graph(t['atoms']).edges )
            
            # this graph gives us the neighbors
            for x in range(self.N):
                self._nn[x] = G.adj[x]
            
        return self._nn
    
    def find_neighbors(self, n, i=None):
        '''
        Constructs the tensor of the n-th nearest neighbor. 
        Careful, this implementation can be ill defined for non-periodic lattices. 
        n : neighbors number we wish to have info on
        i : indicates wether we only want the neighbors of a single particle
            if yes, it indicates which particle
            
        return : list containing the neighbors information
                 i.e. nn[k] = [n-th nearest-neighbors of k]
        '''
        # container of the distances present on the lattice
        # we square it and int it in order to cancel the floating point differences
        neighbors = np.unique(np.rint((self.distances/self.a)**2).reshape(-1), axis=-1)
        
        # the distance we are interested in : the one corresponding to the n-th neighbor
        assert n < len(neighbors), f'There are only {len(neighbors)-1} neighbors, n={n} is too big.'
        d = np.sqrt(neighbors[n]) * self.a
        
        # if we want all neighbors
        if i is None : 
            nn = [[] for k in range(self.N)]

            for i in range(self.N):
                for j in range(self.N):
                    # find which particles are at the corresponding distance
                    if np.isclose(self.distances[i,j], d):
                        nn[i].append( j )
               
        # if we only want the neighbors of i
        else : 
            nn = []
            for j in range(self.N):
                # find which particles are at the corresponding distance
                if np.isclose(self.distances[i,j], d):
                    nn.append( j )
            
        return nn
    
    @property
    def n_distances(self):
        '''
        Number of different distances existing on the lattice (i.e. number of different values in neighbors_distances)

        return : float () with the number of distances 
        '''
        if self._n_distances is None:
            self._n_distances = np.max(self.neighbors_distances)+1
        
        return self._n_distances

    @property
    def neighbors_distances(self):
        '''
        Distance in terms of number of neighbor tensor between the atoms (taking periodiciy into account)
        
        return : ndarray (N,N) containing the neighbor distances d[i,j] = neighbor n
        '''
        if self._neighbors_distances is None:
            # gives the physical distances of neighbor i
            neighbors = np.unique(np.rint((self.distances/self.a)**2).reshape(-1), axis=-1)

            # now the distances of each site pairs
            dist = np.rint((self.distances/self.a)**2)

            # find which it corresponds to to know what neighbor they are
            neighbors_distances = np.zeros((self.N,self.N), dtype=int)
            for i in range(self.N):
                for j in range(self.N):
                    neighbors_distances[i,j] = np.argmin( np.abs(dist[i,j] - neighbors) )
            self._neighbors_distances = neighbors_distances

        return self._neighbors_distances


    @partial(jit, static_argnums=0)
    def dimer_probs(self, samples):
        '''
        Calculates the probability of presence of monomers, single dimers and double dimers etc on the bulk of the lattice (could be up to four)
        !!! WARNING : only works in the Z basis !!!

        samples : a bunch of samples over which to compute the probability (...,N)

        return : array of probabilities p=np.array([p_monomer, p_dimer, p_doubledimer, p_triple, p_quad)]) (5,)
        '''
        # we only consider the vertices which have 4 atoms -> not on the border
        # for a Torus lattice, the border is empty
        n = len(self.vertices) #this container was different for each type of lattice

        # the occupancy of each vertex by sample
        # i.e. number of excited states per vertex
        occupancy = jnp.array([jnp.sum( (1+samples[...,jnp.array(v['atoms'])])/2, axis=-1) for v in self.vertices])

        # find out how many of each configuration is present in total
        p0 = jnp.count_nonzero((occupancy-0)==0)/n # no dimer
        p1 = jnp.count_nonzero((occupancy-1)==0)/n # one dimer
        p2 = jnp.count_nonzero((occupancy-2)==0)/n # two dimers
        p3 = jnp.count_nonzero((occupancy-3)==0)/n # three dimers
        p4 = jnp.count_nonzero((occupancy-4)==0)/n # four dimers

        # combine everything and return it normalized to have a probability
        p = jnp.array([p0, p1, p2, p3, p4])

        return p/p.sum()
    
    