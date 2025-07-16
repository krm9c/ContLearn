#=================FFN MLP with AWB======================#
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
    

#=================CNN with AWB===================#
#note: this CNN works for both standard and AWB training. It can
#be used for both without an issue.
class CNN_AWB(eqx.Module):
    conv_layers: list
    feed_layers: list
    A_conv: jax.Array
    B_conv: jax.Array
    A_feed: jax.Array
    B_feed: jax.Array
    feed_sizes: list
    filter_size: int
    channel_out: int

    def __init__(self, key, filter_size, feed_sizes):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        # Standard CNN setup: convolutional layer, followed by flattening,
        # with a small MLP on top.
        i=0
        self.feed_sizes = feed_sizes
        self.filter_size = filter_size
        self.feed_layers = []
        self.channel_out = 3
        self.conv_layers = [
            eqx.nn.Conv2d(1,self.channel_out, kernel_size=filter_size, key=key1),
            ]
        for (in_layer,out_layer) in zip(feed_sizes[:-1],feed_sizes[1:]):
            self.feed_layers.append(Linear2(in_layer,out_layer, key = jax.random.PRNGKey(i)))
            i+=1
        new_arch = [1875,700,100,10]
        initializer = jax.nn.initializers.glorot_uniform()
        self.A_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(feed_sizes[1:],new_arch[1:])]
        self.B_feed = [initializer(jax.random.PRNGKey(5), (y, x)) for x,y in zip(feed_sizes[:-1],new_arch[:-1])]
        #below should produce the three weights filters
        new_filter_size = 5
        self.B_conv = [jax.random.normal(jax.random.PRNGKey(j),shape = (new_filter_size,filter_size)) for j in range(0,self.channel_out)]
        self.A_conv = [jax.random.normal(jax.random.PRNGKey(j),shape = (new_filter_size,filter_size)) for j in range(0,self.channel_out)]

    def calc_output_size(self,fil_size):
        omni_input = 28
        padding = 0 #this is apparently the default for Conv2d
        stride = 1
        output = ((omni_input-fil_size+2*padding)/stride) + 1
        return int(output)
    
    def pool_output_size(self,pool_size,conv_inputsize):
        stride = 1
        output = ((conv_inputsize-pool_size)/stride) + 1
        return int(output)

    def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]:  
        x = jnp.ravel(jax.nn.relu(eqx.nn.MaxPool2d(kernel_size=2)(self.conv_layers[0](x))))
        for lin in self.feed_layers[:-1]:
            #print("x in model:", x.shape)
            x = jax.nn.relu(lin(x))
        x = self.feed_layers[-1](x)
        #for lin in self.feed_layers:
            #x = jax.nn.relu(lin(x))
        #x = self.feed_layers[0](x)
        return x
        #x = jax.nn.relu(self.feed_layers[0](x))
        #x = jax.nn.relu(self.feed_layers[1](x))
        #x = self.feed_layers[2](x)
        #return x


    def get_AWBT(self,x):
        weights_list = [[(self.A_conv[i]@(self.conv_layers[0].weight[i][0])@jnp.transpose(self.B_conv[i]))] for i in range(0,self.channel_out)]
        #print("weights list shape: ", jnp.array(weights_list).shape)
        x = jnp.expand_dims(x, axis=0)
        #print("x: ", x.shape)
        x = jax.lax.conv_general_dilated(lhs = x, rhs = jnp.array(weights_list), window_strides= (1,1), padding="VALID")
        #print("shape of x after:", x.shape)
        x = x.squeeze(0)
        #print("x: ", x.shape)
        x = jnp.ravel(jax.nn.relu(eqx.nn.MaxPool2d(kernel_size=2)(x)))
        #print("x after: ", x.shape)
        for i in range(0,len(self.feed_sizes)-1):
            #print(x.shape)
            #print("AWBTx: ", (self.A_feed[i] @ self.feed_layers[i].weight @ jnp.transpose(self.B_feed[i]) @ x).shape)
            #print("bias part: ", (self.A_feed[i]@self.feed_layers[i].bias).squeeze(1).shape)
            x = (self.A_feed[i] @ self.feed_layers[i].weight @ jnp.transpose(self.B_feed[i]) @ x) + (self.A_feed[i]@self.feed_layers[i].bias).squeeze(1)
            #print("after: ", x.shape)
            x = jax.nn.relu(x)
            #print(x.shape)
        return x

#----------Linear Layer for CNN_AWB-----------------#
class Linear2(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    initializer: None 
    def __init__(self, in_size, out_size, key):
        self.initializer = jax.nn.initializers.glorot_uniform()
        wkey, bkey = jax.random.split(key)
        self.weight = self.initializer(wkey, (out_size, in_size))
        self.bias = self.initializer(bkey, (out_size,1))
    def __call__(self, x):
        #print("w,x: ", self.weight.shape, x.shape)
        x = self.weight @ x        
        #print("x", x.shape)
        x = x+ self.bias.squeeze(1)
        #print("after bias: ", x.shape)
        return x