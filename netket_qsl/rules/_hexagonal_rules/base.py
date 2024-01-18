import jax 
import jax.numpy as jnp
from jax import jit, vmap

from ...lattice import neighbors


@jit
def _apply(x:jnp.ndarray, sites:jnp.ndarray):
    '''
    Applies the Q transition on multiple sites
    x : sample on which to apply the transition 
    sites : the sites of the path of te operator

    return : changed sample
    '''

    """
    @jit
    def _apply_one_triangle(x:jnp.ndarray, i:int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        '''
        finds the connected element and matrix element of a single element x for an operator acting on site i
        x : flat sample (Ns,N)
        i : single site on which the operator is acting

        return : σ (Ns,N), <x|O|σ> (Ns,)
        '''
        # Define the nn
        j, k = neighbors(i)

        x = jnp.array(x)       

        # which case are we in : s1 s2 = -1 or s1 s2 = 1
        flag = x[j]*x[k]


        # if sj * sk = 1, flip i with amplitude -1
        def flip(x,i,j,k):
            return x.at[i].set( -x[i] )

        # is sj * sk = -1, swap sj and sk with amplitude 1
        def swap(x,i,j,k):
            Exchanged = x.at[j].set( x[k] )
            Exchanged = Exchanged.at[k].set( x[j] )
            return Exchanged

        # the final state is : i flipped if flag==1, jk swaped if flag ==-1
        eta = jax.lax.cond(flag==1, flip, swap, x,i,j,k)

        # the amplitude is always the opposite of the flag
        m = -flag

        return eta, m
    """
    @jit
    def _flip_one_Q_(s, index):
        '''
        Applies the Q at spin index operator on one sample 
        '''
        j,k = neighbors(index)
        s_prime = s.at[index].set( -s[index]*s[j]*s[k] )
        s_prime = s_prime.at[j].set( s[k] )
        s_prime = s_prime.at[k].set( s[j] )

        return s_prime, -s[j]*s[k]

    # loop over the sites, x as input and output, m_primes from each loop
    x_primes, _ = jax.lax.scan(_flip_one_Q_, x, sites)

    # in the end, we return the final x and multiply all the m_els
    return x_primes

@jit
def _do_nothing(x:jnp.ndarray, sites:jnp.ndarray):
    '''
    Does nothing on the sample
    x : sample on which to apply the transition 
    sites : the sites of the path of te operator

    return : unchanged sample
    '''
    return x
    


@jit
def _global_transition(key, σ, hexs):
    return _global_transition_batch(key, σ.reshape(1,-1), hexs).reshape(-1)

@jit
def _global_transition_batch(key, σ, hexs):
    '''
    Defines a transition rule where we choose randomly (uniformly) an hexagon on which to apply Q

    key: The PRNGKey for the random choice
    σ:  initial spin chains

    returns : new chain (...,N)
    '''

    #number of chains
    n_chains = σ.shape[0]
    # number of hexagons
    n_hexs = hexs.shape[0]

    # split the random key
    key, _ = jax.random.split(key,2)

    
    # choose for each hexagon wether or not we apply the string
    # since the mean is at sum p_i = n_hexs p, p = mean/n_hexs = 0.5
    #p = self.mean_global/n_hexs
    # conds = jax.random.choice(key, jnp.array([True,False]), shape=(n_chains,n_hexs) )

    # def _one_transition(conds, s):
    #     '''
    #     travels across all hexagons
    #     for i in range(0,n_hexs):
    #         s = _bodyfun(i,s)
    #     return s
    #     '''

    #     def _body_fun(i, s):
    #         '''
    #         executes one hexagon if condition
    #         if conds[i]:
    #             _apply(s, hexs[i])
    #         else:
    #             _do_nothing(s, hexs[i])
    #         '''
    #         return jax.lax.cond( conds[i], _apply, _do_nothing, s, hexs[i] )

    #     x = jax.lax.fori_loop(0, n_hexs, _body_fun, s)

    #     return x


    # return vmap(_one_transition, in_axes=(0,0))(conds, σ)

    which = jax.random.choice(key,n_hexs,(n_chains,))
    return vmap(_apply, in_axes=(0,0))(σ, hexs[which])
    
