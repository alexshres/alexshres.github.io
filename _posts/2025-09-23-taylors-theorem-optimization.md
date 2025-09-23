## Taylor's Theorem and Optimization

Taylor's theorem is the backbone of numerical optimization and having a solid understanding of its underpinnings can help with intuition. Some of the core concepts that are needed to understand Taylor's Theorem include **Mean Value Theorem (MVT)**, **MVT for Integrals**, and **Fundamental Theorem of Calculus**. The concepts of continuity and differentiability from real analysis are also important. This post will be using Nocedal and Wright's (NW) Numerical Optimization book as source.


#### Things To Know (Single Variable Case)
The following definitions will be in the single-variable case but also apply to the multivariable case.

* **Continuous**: A function \\(f: \mathbb{R} \to \mathbb{R}\\) is continuous in its domain if 

$$
\forall \epsilon > 0, \exists \delta > 0 | |x-c| < \delta \rightarrow |f(x)-f(c)| < \epsilon
$$

* **Differentiable**: A function \\(f: \mathbb{R} \to \mathbb{R}\\) is differentiable if

$$
    f^{\prime}(c) = lim_{x \mapsto c} \frac{f(x) - f(c)}{x-c}
$$

* **Continuously differentiable**: Function is differentiable and it's derivative is continuous

* **Mean Value Theorem (Single Variable)**: \\( \exists c \in \mathbb{R} | f^{\prime}(c) = (f(b)-f(a))(b-a)  \\)
* **Mean Value Theorem for Integrals**: \\(  \exists c \in mathbb{R} | f(c)(b - a) = \int_{a}^{b} f(x) dx   \\)

#### Taylor's Theorem
According to NW, Taylor's Theorem is defined as following:
> Let \\(f: \mathbb{R}^{n} \to \mathbb{R}\\) be continuously differentiable and let \\(p \in \mathbb{R}^{n}\\).
>
> (Eq. 1) $$f(x+p) = f(x) + \nabla f(x+tp)^{T}p \quad \text{for some } t \in (0, 1)$$
>
> If \\(f\\) is **twice** continuously differentiable, then we have the following results:
>
> (Eq. 2) $$\nabla f(x+p) = \nabla f(x) + \int_{0}^{1}\nabla^{2}f(x+tp)p \, dt$$
>
> and
>
> (Eq. 3) $$f(x+p) = f(x) + \nabla f(x)^{T}p + \frac{1}{2}p^{T}\nabla^{2} f(x+tp)p, \quad \text{for some } t \in (0, 1)$$

We will go over how each of these are derived.


Starting with (Eq. 1).

Notes:
* start with \\(g(t) = f((1-t)x+ta)\\) line segment between x and a where t parameterizes a point on that line segment
* note \\(g(0) = f(x)\\) and \\(g(1) = f(a)\\)
* Mean value theorem for 1 variable states there exists some \\(g^{\prime}(c) = g(1)-g(0)(1-0)\\) for some \\(c \in (0, 1)\\)
* \\(g^{\prime}(t) = \nabla f((1-t)x+sa)^{T}(a-x)\\) NTK that we are dealing with multivariable inputs so derivative needs to be taken accordingly
* plugging everything in: \\(g(1) - g(0) = g^{\prime}(t)(a-x)\\)
* \\(f(a) - f(x) = \nabla f((1-t)x+ta)^{T}(a-x)\\)
* Let \\(a = x+p \\) p is how much we add to get to the second point in our line segment
* \\(f(x+p)-f(x) = \nabla f((1-t)x +t(x+p))^{T}(x+p-x)\\)
* \\(f(x+p) = f(x) + \nabla f(x-tx + tx +tp)^{T}p\\)
* \\(f(x+p) = f(x) + \nabla f(x+tp)^{T}p\\)
* that's our first equation
