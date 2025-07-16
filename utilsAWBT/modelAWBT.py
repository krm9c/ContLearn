#-----------FFN MLP with AWB-----------------#
#note: this MLP works for both standard and AWB training. It can
#be used for both without an issue.
class MLP_AWB(eqx.Module):
    layers: list
    sizes: list
    #act_fn: Callable
    #act_options: list
    A: jax.Array
    B: jax.Array
    
    def __init__(self, sizes):
        self.layers = []
        #self.act_fn = act_fn
        #self.act_options = act_options
        self.A = [jax.random.normal(jax.random.PRNGKey(0),shape = (y,1)) for y in sizes[1:]]
        self.B = [jax.random.normal(jax.random.PRNGKey(0),shape = (y,x)) for x,y in zip(sizes[:-1],sizes[1:])]
        i = 0
        self.sizes  = sizes
        for (in_layer,out_layer) in zip(sizes[:-1],sizes[1:]):
            self.layers.append(Linear_AWB(in_layer,out_layer, key = jax.random.PRNGKey(i)))
            i+=1

    def __call__(self, x):
        for lin in self.layers[:-1]:
            x = jax.nn.tanh(lin(x))
        x = self.layers[-1](x)
        return x

    def getAWB(self,x):
        for i in range(0,len(self.sizes)-1):
            #x=(model.A[i] @ model.layers[i].weight @ jnp.transpose(model.B[i]))
            #x = (self.A[i] @ self.layers[i].weight @ jnp.transpose(self.B[i]) @ x) + (self.A[i] @ self.layers[i].bias)
            #print("this is the shape of AWB.T each time: ", (self.A[i] @ self.layers[i].weight @ jnp.transpose(self.B[i])).shape)
            #print("this is the shape of AWB.Tx each time: ", (self.A[i] @ self.layers[i].weight @ jnp.transpose(self.B[i]) @ x).shape)
            #print("this is the shape of new bias each time: ", ((self.layers[i].bias @ self.A[i].T).T.squeeze(1)).shape)
            x = (self.A[i] @ self.layers[i].weight @ jnp.transpose(self.B[i]) @ x) + (self.layers[i].bias @ self.A[i].T).T.squeeze(1)
            x = jax.nn.tanh(x)
            #print("this is the shape of x each time: ", x.shape)
        return x
    
#--------Linear Layer for MLP_AWB--------------------#
class Linear_AWB(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    initializer: None 
    def __init__(self, in_size, out_size, key):
        self.initializer = jax.nn.initializers.glorot_uniform()
        wkey, bkey = jax.random.split(key)
        self.weight = self.initializer(wkey, (out_size, in_size))
        self.bias = self.initializer(bkey, (1, out_size))
    def __call__(self, x):
        # print(self.weight.shape, x.shape)
        x = x @ self.weight.T
        # print(x.shape)
        x = x+ self.bias
        # print(x.shape)
        return x