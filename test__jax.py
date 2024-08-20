import jax
import jax.numpy as jnp


@jax.jit
def func(x, w, phi):
    return jnp.linalg.norm(x-jax.lax.stop_gradient(w) )**2+jnp.linalg.norm(phi)**2
x=0.1
w=0.2
phi =0.4
# print(func(x, w, phi))
xdot = 0.01
wdot=0.02
phidot = 0.04
j, jdot = jax.jvp(func, (x, w, phi), (xdot, wdot, phidot)) 
print(j, jdot )
primals, func__jvp = jax.linearize(func, x, w, phi)
print(primals, func__jvp( xdot, wdot, phidot ) )
grad = jax.grad(func__jvp, argnums=(2))(x, w, phi)+ jax.grad(func, argnums=(1))(x, w, phi)
print(grad)