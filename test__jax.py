# import jax
# import jax.numpy as jnp


# @jax.jit
# def func(x, w, phi):
#     return jnp.linalg.norm(x-jax.lax.stop_gradient(w) )**2+jnp.linalg.norm(phi)**2
# x=0.1
# w=0.2
# phi =0.4
# # print(func(x, w, phi))
# xdot = 0.01
# wdot=0.02
# phidot = 0.04
# j, jdot = jax.jvp(func, (x, w, phi), (xdot, wdot, phidot)) 
# print(j, jdot )
# primals, func__jvp = jax.linearize(func, x, w, phi)
# print(primals, func__jvp( xdot, wdot, phidot ) )
# grad = jax.grad(func__jvp, argnums=(2))(x, w, phi)+ jax.grad(func, argnums=(1))(x, w, phi)
# print(grad)


import jax
from jax import numpy as jnp
import equinox as eqx
from typing import Callable

class NN(eqx.Module):
  w: jax.Array
  b: jax.Array
  act_fn: Callable

  def __call__(self, x):
    return self.act_fn(self.w @ x + self.b)

def lin():

  full_net = NN(jnp.ones((10, 10)), jnp.ones(10), jax.nn.relu)
  net, static = eqx.partition(full_net, eqx.is_array)

  def f(inputs):
    a, b = inputs
    return eqx.combine(a, static)(b)

  y, f_jvp = jax.linearize(f, (net, jnp.ones(10)))
  
  out_tangent = f_jvp((net, 0.1 * jnp.ones(10)))


  def f(inputs):
    a, b = inputs
    return a(b)

  y1, out_tangent1 = eqx.filter_jvp(f, primals=[(full_net, jnp.ones(10))],
                                             tangents=[(net, 0.1 * jnp.ones(10))])

  assert jnp.allclose(y, y1)
  assert jnp.allclose(out_tangent, out_tangent1)

lin()
