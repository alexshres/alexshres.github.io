## Taylor's Theorem and Optimization

Taylor's theorem is the backbone of numerical optimization and having a solid understanding of its underpinnings can help with intuition. Some of the core concepts that are needed to understand Taylor's Theorem include **Mean Value Theorem (MVT)**, **MVT for Integrals**, and **Fundamental Theorem of Calculus**. The concepts of continuity and differentiability from real analysis are also important.


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







$$
f(x+p) = f(x) + \nabla f(x)^{T}p + \frac{1}{2}p^{T}\nabla^{2}f(x+tp)p
$$