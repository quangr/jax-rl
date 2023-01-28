怎么在jax和flax中做线性回归

```
from flax import linen as nn
import jax.random as random
```
这两个库相当于，torch.nn和numpy.random。
```
model = nn.Dense(features=1)
```
相当于在torch中，使用`torch.nn.Linear(n,1)`,这里不用指定输入维度，是因为我们在后面的model.init中可以推断出输入的维度。

```
key=random.PRNGKey(0)
key1, key2 = random.split(key)
x = random.normal(key1, (100,1,)) 
```
一个典型的伪随机数是迭代生成的。

jax中的随机数和numpy中的随机数区别是jax中的random函数是无状态的，所以每次执行都需要一个key参数，由于纯函数的性质，给定相同的key和形状参数，生成的随机数都是相同的。我们可以通过random.split不停的递归生成新的key。


```
params = model.init(key2, x) # Initialization call
jax.tree_util.tree_map(lambda x: x.shape, params)

```
注意到模型的初始化除了需要一个key来随机生成参数以外，还需要一个dummy input来让flax推断出每个层的形状。init函数返回的是初始化参数。

注意如果在上面我们使用的是`x = random.normal(key1, (323,23,1,))`,生成的神经网络形状是一样的，而使用`x = random.normal(key1, (23,10)) `的话，会生成一个由10维映射到一维的神经网络层，这是因为nn.Dense推断规则是根据输入数据的最后一层来确定的。我们可以使用`model.apply(params, x)`来获取模型在params参数下，输入x的取值。

返回的params本质上是一个pynodetree，也就是把python对象看作一个树，那么所有的numpy对象可以看作一个叶子，用这个方法来刻画一个储存参数的对象。我们可以用jax.tree_util.tree_map来对每一个叶子节点进行映射，jax.tree_util.tree_map的第一个参数是函数，返回的结果是将所有叶子节点进行映射后得到的树。

```
import jax.numpy as jnp
def mse(params,x,y):
  newy=model.apply(params,x)
  return ((newy-y)**2).mean()
```
这段代码定义了mse函数，即训练时的损失函数。有了损失函数以后，我们可以用jax.value_and_grad把mse函数转化为一个能返回求值和倒数的函数。

```
learning_rate = 0.3  # Gradient step size.
loss_grad_fn = jax.value_and_grad(mse)

loss_val, grads = loss_grad_fn(params, x, 2*x+3)
```
注意这时候转化后的`loss_grad_fn`函数除了返回取以外，还会返回梯度。注意这个grads的结构和params是一模一样的。

```
@jax.jit
def update_params(params, learning_rate, grads):
  params = jax.tree_util.tree_map(
      lambda p, g: p - learning_rate * g, params, grads)
  return params

for i in range(101):
  # Perform one gradient update.
  loss_val, grads = loss_grad_fn(params, x_samples, y_samples)
  params = update_params(params, learning_rate, grads)
  if i % 10 == 0:
    print(f'Loss step {i}: ', loss_val)
```
得到梯度以后，我们怎么样才能在jax中做梯度下降呢？我们可以直接用jax.tree_util.tree_map做映射。返回就是新的参数了。



```
import optax
tx = optax.adam(learning_rate=learning_rate)
opt_state = tx.init(params)
loss_grad_fn = jax.value_and_grad(mse)
for i in range(101):
  loss_val, grads = loss_grad_fn(params, x_samples, y_samples)
  updates, opt_state = tx.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  if i % 10 == 0:
    print('Loss step {}: '.format(i), loss_val)
```
除了做最简单的梯度下降以外，我们还可以用optax库中使用一些已经封装好的优化器，比如adam优化器。我们可以用init方法获取adam优化器的初始状态（所有的有状态过程在jax中都会以某种变量保存在函数外部）。然后使用update获取经过计算的grads。

```
from flax.training.train_state import TrainState
state=TrainState.create(apply_fn=None,params=params,tx=optax.adam(learning_rate=learning_rate))
```
一种更优雅的方法使用flax提供的TrainState，将参数和优化器包装在一起。这种情况下，我们可以这样写训练过程。
```
for i in range(100):
  loss_val, grads = loss_grad_fn(state.params, x, 2*x+3)
  state = state.apply_gradients(grads=grads)
  if i % 10 == 0:
    print('Loss step {}: '.format(i), loss_val)

```
