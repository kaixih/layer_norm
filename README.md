# Layer Normalization Backprop
## Layer Normalization Forward Pass
Suppose $N, D$ are batch and feature dimensions respectively.
<img src="https://render.githubusercontent.com/render/math?math=x: (N, D)"> 
$$x: (N, D)$$
$$\mu, \sigma ^2: (N, 1)$$
$$\gamma, \beta: (1, D)$$
$$\mu = \frac{1}{D}\sum_{i=1}^{D}x_i$$
$$\sigma^2 = \frac{1}{D}\sum_{i=1}^{D}(x_i - \mu)^2$$
$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2+\epsilon}}$$
$$y=\gamma\hat{x} + \beta$$

## Layer Normalization Backward Pass
$$dy: (N, D)$$
$$d\mu, d\sigma ^2: (N, 1)$$
$$d\gamma, d\beta: (1, D)$$

$$d\gamma=dy\cdot \frac{\partial y}{\partial \gamma}=\sum_{i=1}^{N}dy\cdot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}$$

$$d\beta=dy\cdot \frac{\partial y}{\partial \beta}=\sum_{i=1}^{N}dy$$

Here we can view $y=\gamma\hat{x} + \beta$ as a composition function $y(i(x), \sigma^2(x), \mu(x))$, where $i(x)=x$ and $\sigma^2(x)$, $\mu(x)$ are defined as above. Therefore, we get:
$$dx = \frac{\partial L}{\partial i}\cdot\frac{\partial i}{\partial x} + \frac{\partial L}{\partial \sigma^2}\cdot\frac{\partial \sigma^2}{\partial x} + \frac{\partial L}{\partial \mu}\cdot\frac{\partial \mu}{\partial x}$$

* Compute $\frac{\partial L}{\partial i}\cdot\frac{\partial i}{\partial x}$
$$\frac{\partial L}{\partial i} = \frac{\partial L}{\partial y}\cdot\frac{\partial y}{\partial i} = dy \cdot \gamma \cdot \frac{1}{\sqrt{\sigma^2+\epsilon}}$$
$$\frac{\partial i}{\partial x}=1$$

* Compute $\frac{\partial L}{\partial \sigma^2}\cdot\frac{\partial \sigma^2}{\partial x}$
$$\frac{\partial L}{\partial \sigma^2} = \frac{\partial L}{\partial y}\cdot\frac{\partial y}{\partial \sigma^2} = \sum_{i=1}^{D}dy \cdot \gamma (x-\mu)(-\frac{1}{2})(\sigma^2+\epsilon)^{-\frac{3}{2}}$$
$$\frac{\partial \sigma^2}{\partial x}=\frac{2}{D}(x-\mu)$$

* Compute $\frac{\partial L}{\partial \mu}\cdot\frac{\partial \mu}{\partial x}$
$$\frac{\partial L}{\partial \mu}=\frac{\partial L}{\partial \mu}\cdot \frac{\partial \mu}{\partial \mu} + \frac{\partial L}{\partial \sigma^2}\cdot \frac{\partial \sigma^2}{\partial \mu}\\=\frac{\partial L}{\partial y}\cdot \frac{\partial y}{\partial \mu}\cdot 1 + \frac{\partial L}{\partial \sigma^2}\cdot \frac{\partial \sigma^2}{\partial \mu}\\=\sum_{i=1}^{D}dy\cdot \gamma \cdot\frac{-1}{\sqrt{\sigma^2+\epsilon}}+\sum_{i=1}^{D}\frac{\partial L}{\partial \sigma^2}\cdot \frac{1}{D}(-2)(x-\mu)$$
$$\frac{\partial \mu}{\partial x}=\frac{1}{D}$$
