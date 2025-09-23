## Taylor's Theorem and Optimization

Taylor's theorem is the backbone of numerical optimization and having a solid understanding of its underpinnings can help with intuition. Some of the core concepts that are needed to understand Taylor's Theorem include **Mean Value Theorem (MVT)**, **MVT for Integrals**, and **Fundamental Theorem of Calculus**. The concepts of continuity and differentiability from real analysis are also important. This post will be using Nocedal and Wright's (NW) Numerical Optimization book as source.


#### Things To Know (Single Variable Case)
The following definitions will be in the single-variable case but also apply to the multivariable case.

* **Continuous**: A function $f: \mathcal{R} \to \mathcal{R}$ is continuous in its domain if 

$$
\forall \epsilon > 0, \exists \delta > 0 | |x-c| < \delta \rightarrow |f(x)-f(c)| < \epsilon
$$

* **Differentiable**" A function $f: \mathcal{R} \to \mathcal{R}$ is differentialbe if

$$
    f^{\prime}(c) = lim_{x \mapsto c} \frac{f(x) - f(c)}{x-c}
$$

* **Continuously differentiable**: Function is differentiable and it's derivative is continuous


#### Taylor's Theorem
According to NW, Taylor's theorem is defined as following:
> Let $f: \mathbb{R}^{n} \to \mathbb{R}$ be continuously differentiable and let $p \in \mathbb{R}^{n}$.
>
> (Eq. 1) $$f(x+p) = f(x) + \nabla f(x)^{T}p + \frac{1}{2}p^{T}\nabla^{2}f(x+tp)p, \quad \text{for some } t \in (0, 1)$$
>
> If $f$ is **twice** continuously differentiable, then we have the following results:
>
> (Eq. 2) $$\nabla f(x+p) = \nabla f(x) + \int_{0}^{1}\nabla^{2}f(x+tp)p \, dt$$
>
> and
>
> (Eq. 3) $$f(x+p) = f(x) + \nabla f(x)^{T}p + \frac{1}{2}p^{T}\nabla^{2} f(x+tp)p, \quad \text{for some } t \in (0, 1)$$

> Let $f: \mathcal{R}^{n} \to \mathcal{R}$ be continuously differentiable and $p \in \mathcal{R}^{n}$. Then we have that:
> (Eq. 1) $$f(x+p) = f(x) + \nabla f(x)^{T}p + \frac{1}{2}p^{T}\nabla^{2}f(x+tp)p \text{\tab For some $t \in (0, 1)$} $$
> If $f$ is twice continuously differentiable then we have 
> (Eq. 2) $$\nabla f(x+p) = \nabla f(x) + \int_{0}^{1}\nabla^{2}f(x+tp)p dt$$
> and that
> (Eq. 3) $$f(x+p) = f(x) + \nabla f(x)^{T}p + \frac{1}{2}p^{T}\nabla^{2} f(x+tp)p \text{\tab for some $t \in (0, 1)$} $$

We will go over how each of these are derived.








$$
$$