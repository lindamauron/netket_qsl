import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap, jit
from ..hilbert import TriangleHilbertSpace
import flax.linen as nn
from netket.utils.types import DType, Array, NNInitFunc


import os.path as path
text_file_path = path.dirname(path.abspath(__file__)) + '/indices.npy'
bare_number = np.load(text_file_path)


@jit
def states_to_numbers(states):
    #states = states.reshape(-1,states.shape[-1])
    coeffs = (states+1)/2
    unconst_n = (coeffs*2**jnp.arange(24)[::-1]).sum(-1).astype(int)
    
    return jnp.searchsorted(bare_number, unconst_n)


@jit
def angular_apply_on_ggg(s, coeff, r1,r2,t1,t2, t):
    """
    Applies the RVB ansatz operator on one triangle of s with no excitation : O_t (coeff |...ggg...>)
    s : state
    coeff : coefficient of the basis state s
    r1,r2,t1,t2 : parameters of the model
    t : triangle
    
    returns : matrix elements (4,) and connected elements (4,24)
    """

    # convert the parameters
    c1 = jnp.cos(r1)
    s1 = jnp.exp(1j*t1)*jnp.sin(r1)
    c2 = jnp.cos(r2)
    s2 = jnp.exp(1j*t2)*jnp.sin(r2)

    # first, the one that doesnt change
    m0 = c1**3*c2**3
    s0 = jnp.array(s)

    # putting an excitation always with same matrix element
    mi = c1**3*c2**2*s2
    s1 = s0.at[3*t].set(1)
    s2 = s0.at[3*t+1].set(1)
    s3 = s0.at[3*t+2].set(1)

    # construct all coefficients and states obtained
    ms = coeff*jnp.array([m0,mi,mi,mi])
    sigmas = jnp.array([s0,s1,s2,s3])
    return ms, sigmas

@jit
def angular_apply_on_ggr(s, coeff, r1,r2,t1,t2, t):
    """
    Applies the RVB ansatz operator on one triangle of s with one excitation : O_t (coeff |...ggr...>) or permutation
    s : state
    coeff : coefficient of the basis state s
    r1,r2,t1,t2 : parameters of the model
    t : triangle
    
    returns : matrix elements (4,) and connected elements (4,24)
    """
    
    # convert the parameters
    c1 = jnp.cos(r1)
    s1 = jnp.exp(1j*t1)*jnp.sin(r1)
    c2 = jnp.cos(r2)
    s2 = jnp.exp(1j*t2)*jnp.sin(r2)

    # i is the (only) place where s[i]=1, so we construct it by calculation to avoid a bool (1+s_i)/2 <=> bool(s_i==1)
    i = (3*t +  ( 0*(1+s[3*t]) + 1*(1+s[3*t+1]) + 2*(1+s[3*t+2]) )/2 ).astype(int)

    # j,k are the two other numbers in the triangle
    j = (i+1)%3 + 3*t
    k = (i+2)%3 + 3*t

    # first si gets no change
    si = jnp.array(s)
    mi = c1**2*c2**2*(c1*c2+s1*s2)

    # the one projected on |ggg>
    s0 = si.at[i].set(-1)
    m0 = c1**2*s1*c2**3

    # the last ones are the swaps si<->sj and si<->sk
    sj = s0.at[j].set(1)
    sk = s0.at[k].set(1)
    mjk = c1**2*c2**2*s1*s2

    # construct all coefficients and states obtained
    ms = coeff*jnp.array([m0,mi,mjk,mjk])
    sigmas = jnp.array([s0,si,sj,sk])
    return ms, sigmas


class RVBProj_angular(nn.Module):
    r"""
    Ansatz inspired from arXiv:2201.04034v1 consisting of \psi(\sigma) = <\sigma|\prod_i (cos(r2) 1 + sin(r2)e^{it2} \sigma_i^+)(cos(r1) 1 + sin(r1)e^{it1} \sigma_i^-)|RVB>
    The RVB state is an equal weight superposition of all the maximal dimer coverings of the kagome lattice Torus(a,2,4). 
    This model is then only implemented on the toy model mentioned since one needs to exactly know the RVB state. This instance works only on the restricted hilbert space. 
    """

    hilbert:TriangleHilbertSpace
    """The Hilbert space."""
    #all_states:jnp.ndarray = jnp.zeros((65000,24))

    param_dtype: DType = jnp.float64
    """The dtype of the weights."""

    param_init: NNInitFunc = nn.initializers.ones
    """Initializer for the weights."""
    
    def setup(self):
        """
        Defines all quantities necessary for the call of the state. 
        """
        if not self.hilbert.n_states == 65536:
            raise ValueError(
                'You can only use this model on the toy model defined on Torus(1.0,2,4)'
            )
        # Declare all the parameters
        self.r1 = self.param( 'r1', self.param_init, (1,), self.param_dtype)
        self.r2 = self.param( 'r2', self.param_init, (1,), self.param_dtype)
        self.t1 = self.param( 't1', self.param_init, (1,), self.param_dtype)
        self.t2 = self.param( 't2', self.param_init, (1,), self.param_dtype)
        
        # the rvb state, whose components have been found previously
        self.rvb = jnp.zeros((65536,), dtype=jnp.complex128).at[jnp.array([4005,4090,5657,5702,10043,10084,16007,16088,
                                                                           18713,18758,20645,20730,24967,25048,30779,30820,
                                                                           36241,36302,37933,38002,42255,42320,48307,48364,
                                                                           52013,52082,53905,53966,58291,58348,64015,64080])].set(1/jnp.sqrt(32))
        # store the basis states since we will need them at each call
        self.all_states = jnp.array( self.hilbert.all_states() )
        

    def __call__(self, x_in: Array):
        """
        returns the log \psi(x_in)
        """
        
        @jit
        def body_fun(t, state):
            """
            applies the operator acting on one triangle on state
            t : index of the triangle [0,7]
            state : input state on which to apply the operator
            """


            ## find connected elements
            
            # this tells us whether we have an excitation in the triangle t or not
            conds = jnp.prod(self.all_states[:,3*t+jnp.arange(3)], axis=-1)+1
            # gets the connected element (for all states) when applying the operator to the triangle
            new_ms, sigmas = vmap(jax.lax.cond, in_axes=(0,None,None,0,0,None,None,None,None,None))( conds, angular_apply_on_ggr, angular_apply_on_ggg, 
                                                                                      self.all_states, state,self.r1,self.r2,self.t1,self.t2, t)
            # find which states appeared
            s_indices = states_to_numbers(sigmas)
            
            # loop over all states of the basis to know what each one changed to the global state
            @jit
            def f(k, state):
                """
                adds the k-th components to the already existing state
                k : component of the basis to consider [0,65536]
                state : input state on which to add the components
                """
                return state.at[s_indices[k]].set( state[s_indices[k]] + new_ms[k,:,0] )
            
            return jax.lax.fori_loop(0, 65536, f, jnp.zeros((65536,), dtype=jnp.complex128))

        # loop on all triangles
        output = jax.lax.fori_loop(0, 8, body_fun, self.rvb)

        # now find the overlap with the input and take the log
        return jnp.log( output[states_to_numbers(x_in)] )


@jit
def normal_apply_on_ggg(s, coeff, z1, z2, t):
    """
    Applies the RVB ansatz operator on one triangle of s with no excitation : O_t (coeff |...ggg...>)
    s : state
    coeff : coefficient of the basis state s
    z1,z2 : parameters of the model
    t : triangle
    
    returns : matrix elements (4,) and connected elements (4,24)
    """

    # first, the one that doesnt change
    m0 = 1
    s0 = jnp.array(s)

    # putting an excitation always with same matrix element
    mi = z2
    s1 = s0.at[3*t].set(1)
    s2 = s0.at[3*t+1].set(1)
    s3 = s0.at[3*t+2].set(1)

    # construct all coefficients and states obtained
    ms = coeff*jnp.array([m0,mi,mi,mi])
    sigmas = jnp.array([s0,s1,s2,s3])
    return ms, sigmas

@jit
def normal_apply_on_ggr(s, coeff, z1, z2, t):
    """
    Applies the RVB ansatz operator on one triangle of s with one excitation : O_t (coeff |...ggr...>) or permutation
    s : state
    coeff : coefficient of the basis state s
    z1,z2 : parameters of the model
    t : triangle
    
    returns : matrix elements (4,) and connected elements (4,24)
    """

    # i is the (only) place where s[i]=1, so we construct it by calculation to avoid a bool (1+s_i)/2 <=> bool(s_i==1)
    i = (3*t +  ( 0*(1+s[3*t]) + 1*(1+s[3*t+1]) + 2*(1+s[3*t+2]) )/2 ).astype(int)

    # j,k are the two other numbers in the triangle
    j = (i+1)%3 + 3*t
    k = (i+2)%3 + 3*t

    # first si gets no change
    si = jnp.array(s)
    mi = 1+z1*z2

    # the one projected on |ggg>
    s0 = si.at[i].set(-1)
    m0 = z1

    # the last ones are the swaps si<->sj and si<->sk
    sj = s0.at[j].set(1)
    sk = s0.at[k].set(1)
    mjk = z1*z2

    # construct all coefficients and states obtained
    ms = coeff*jnp.array([m0,mi,mjk,mjk])
    sigmas = jnp.array([s0,si,sj,sk])
    return ms, sigmas


class RVBProj_normal(nn.Module):
    r"""
    Ansatz inspired from arXiv:2201.04034v1 consisting of \psi(\sigma) = <\sigma|\prod_i (cos(r2) 1 + sin(r2)e^{it2} \sigma_i^+)(cos(r1) 1 + sin(r1)e^{it1} \sigma_i^-)|RVB>
    The RVB state is an equal weight superposition of all the maximal dimer coverings of the kagome lattice Torus(a,2,4). 
    This model is then only implemented on the toy model mentioned since one needs to exactly know the RVB state. This instance works only on the restricted hilbert space. 
    """

    hilbert:TriangleHilbertSpace
    """The Hilbert space."""
    #all_states:jnp.ndarray = jnp.zeros((65000,24))

    param_dtype: DType = jnp.complex128
    """The dtype of the weights."""

    param_init: NNInitFunc = nn.initializers.ones
    """Initializer for the weights."""
    
    def setup(self):
        """
        Defines all quantities necessary for the call of the state. 
        """
        
        if not self.hilbert.n_states == 65536:
            raise ValueError(
                'You can only use this model on the toy model defined on Torus(1.0,2,4)'
            )
        
        # Declare all the parameters
        self.z1 = self.param( 'z1', self.param_init, (1,), self.param_dtype)
        self.z2 = self.param( 'z2', self.param_init, (1,), self.param_dtype)
        
        # the rvb state, whose components have been found previously
        self.rvb = jnp.zeros((65536,), dtype=jnp.complex128).at[jnp.array([4005,4090,5657,5702,10043,10084,16007,16088,
                                                                           18713,18758,20645,20730,24967,25048,30779,30820,
                                                                           36241,36302,37933,38002,42255,42320,48307,48364,
                                                                           52013,52082,53905,53966,58291,58348,64015,64080])].set(1/jnp.sqrt(32))
        # store the basis states since we will need them at each call
        self.all_states = jnp.array( self.hilbert.all_states() )
        

    def __call__(self, x_in: Array):
        """
        returns the log \psi(x_in)
        """
        
        @jit
        def body_fun(t, state):
            """
            applies the operator acting on one triangle on state
            t : index of the triangle [0,7]
            state : input state on which to apply the operator
            """


            ## find connected elements
            
            # this tells us whether we have an excitation in the triangle t or not
            conds = jnp.prod(self.all_states[:,3*t+jnp.arange(3)], axis=-1)+1
            # gets the connected element (for all states) when applying the operator to the triangle
            new_ms, sigmas = vmap(jax.lax.cond, in_axes=(0,None,None,0,0,None,None,None))( conds, normal_apply_on_ggr, normal_apply_on_ggg,
                                                                                          self.all_states, state,self.z1,self.z2, t)
            # find which states appeared
            s_indices = states_to_numbers(sigmas)
            
            # loop over all states of the basis to know what each one changed to the global state
            @jit
            def f(k, state):
                """
                adds the k-th components to the already existing state
                k : component of the basis to consider [0,65536]
                state : input state on which to add the components
                """
                return state.at[s_indices[k]].set( state[s_indices[k]] + new_ms[k,:,0] )
            
            return jax.lax.fori_loop(0, 65536, f, jnp.zeros((65536,), dtype=jnp.complex128))

        # loop on all triangles
        output = jax.lax.fori_loop(0, 8, body_fun, self.rvb)

        # now find the overlap with the input and take the log
        return jnp.log( output[states_to_numbers(x_in)] )
    
class LogStateVector(nn.Module):
    r"""
    _Exact_ ansatz storing the logarithm of the full, exponentially large
    wavefunction coefficients. As with other models, coefficients do not need
    to be normalised.

    This ansatz can only be used with Hilbert spaces which are small enough to
    be indexable.

    By default it initialises as a uniform state.
    """
    hilbert:TriangleHilbertSpace
    """The Hilbert space."""

    param_dtype: DType = jnp.complex128
    """The dtype of the weights."""

    logstate_init: NNInitFunc = nn.initializers.ones
    """Initializer for the weights."""

    def setup(self):
        if not self.hilbert.n_states == 65536:
            raise ValueError(
                'You can only use this model on the toy model defined on Torus(1.0,2,4)'
            )

        self.logstate = self.param(
            "logstate", self.logstate_init, (65536,), self.param_dtype
        )

    def __call__(self, x_in: Array):
        return self.logstate[states_to_numbers(x_in)]