import numpy as np


equiv = {}

equiv['I'] = {}
equiv['X'] = {}
equiv['Y'] = {}
equiv['Z'] = {}

#identities
equiv['I']['X'] = ('X', 1)
equiv['I']['Y'] = ('Y', 1)
equiv['I']['Z'] = ('Z', 1)
equiv['X']['I'] = ('X', 1)
equiv['Y']['I'] = ('Y', 1)
equiv['Z']['I'] = ('Z', 1)

# XZ
equiv['X']['Z'] = ('Y', -1j)
equiv['Z']['X'] = ('Y', 1j)
#XY
equiv['X']['Y'] = ('Z', 1j)
equiv['Y']['X'] = ('Z', -1j)
#YZ
equiv['Y']['Z'] = ('X', 1j)
equiv['Z']['Y'] = ('X', -1j)
#squares
equiv['I']['I'] = ('I', 1)
equiv['X']['X'] = ('I', 1)
equiv['Y']['Y'] = ('I', 1)
equiv['Z']['Z'] = ('I', 1)




def simplify( Operator ):
    ops, w = Operator
    ops = np.array(ops)
    w = np.array(w)
    
    zeros = np.where( w == 0)
    ops = np.delete(ops, zeros)
    w = np.delete(w, zeros)
    
    for k in range(len(ops)):
        i = np.where(ops[k+1:] == ops[k])[0]

        ops = np.delete(ops, i+k+1)
        w[k] += np.sum([w[j] for j in i+k+1])
        w = np.delete(w, i+k+1)
        

        if k == len(ops)-1:
            return ops, w
        
        
def multiplication( Operator1, Operator2 ):
    assert Operator1.hi == Operator2.hi, "Both operators must act on the same hilbert space"
    ops1, w1 = simplify(Operator1)
    ops2, w2 = simplify(Operator2)
    
    n1 = len(ops1)
    n2 = len(ops2)

    new_ops = []
    new_ws = []


    # do for each operator of both lists
    n1 = len(ops1)
    n2 = len(ops2)
    for k1 in range(n1):
        for k2 in range(n2):

            new_op = ''
            new_w = w1[k1]*w2[k2]
            for i in range(Operator1.hi.size):
                s, m = equiv[ops1[k1][i]][ops2[k2][i]]

                new_op = np.char.add(new_op, s)
                new_w *= m
            new_ops.append(new_op)
            new_ws.append(new_w)
            
    return simplify( (new_ops, new_ws) )
    
        
    
def addition( Operator1, Operator2 ):
    ops1, w1 = Operator1
    ops2, w2 = Operator2
    return simplify( (np.concatenate((ops1,ops2)), np.concatenate((w1,w2))) )

