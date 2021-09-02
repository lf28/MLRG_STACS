### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 71d93c0d-0e4a-4684-8a54-6e25c2606a2b
begin
	cd(@__DIR__)
	using Pkg
	Pkg.activate(".")
	Pkg.add("StatsPlots")
	using Distributions, Random, MCMCChains, Plots
	plotly()
	using StatsPlots
	using LinearAlgebra
end

# ╔═╡ fef01c13-fc9e-45f7-9959-0a740f27b458
md"""
## Some quick questions to think about

1. When shall we use MCMC ? What are the assumptions or the problem setting that MCMC is appropriate ? 

2. What Markov Chain mean in MCMC? What is Monte Carlo ?


3. How do you estimate $\pi$ by using a uniform distributed random variable $\boldsymbol{x} \in [0,1]^2$ ? Is this a MCMC method ?


4. To make Bayesian inference on a logistic regression model; what is the model ? what inference options do you we have ? Do we need to use MCMC ?

"""

# ╔═╡ 797629b6-0669-11ec-3778-d1f20c3d6a4c
md"""# MCMC: exercises
- Lei Fang 25/08/2021

Here are a few exercises you can try to improve your understanding of Bayesian inference and MCMC. As most of the problems listed here have no closed-form posterior distributions, you need to resort to MCMC to approximate the true posteriors. You are recommended to do the exercises in the given order.
"""

# ╔═╡ 4570ae10-8419-4af4-aea6-c5fc05756617
md"""
### Question 1 (Conjugate inference)
You are given data set $D_1$, a sample of n=100 i.i.d (independently and identically distributed) observations from a Poisson distribution. A Poisson distribution has a probability mass function 

$$P(X=k|\lambda) = \frac{\lambda^k e^{-\lambda}}{k!},$$ where $\lambda >0$ is the mean of the distribution. Make a Bayesian inference over $\lambda$ given $D_1$. 

!!! hint
    The conjugate prior for $\lambda$ is a Gamma distribution: $$P(\lambda) = \text{Gamma}(a_0,b_0) = \frac{{b_0}^{a_0}}{\Gamma(a_0)} \lambda^{a_0-1} e^{-b_0\lambda}.$$ Show the posterior is still a Gamma distribution with an updated parameters $a_n, b_n$: i.e. $$P(\lambda|D) = \text{Gamma}(a_n, b_n),$$ where $a_n= a_0 + \sum_{i=1}^n d_i, b_n = b_0 +n.$ Then sample from the posterior.

- what the effect of observation size $n$ has on the posterior ?
- how to interpret the prior hyperparameters $a_0$ and $b_0$ (assuming Gamma is used as the prior), i.e.  $$P(\lambda) = \text{Gamma}(a_0,b_0)$$?
"""

# ╔═╡ 13cb28fb-6b72-4ccf-9a2e-aec107ccd04b
begin
	trueλ1 = 10.
	n1 = 100
	D1 = rand(Poisson(trueλ1), n1)
end

# ╔═╡ 92d4bbd4-a7c1-426a-8c26-a40a4cb37cff
md"""**Solution** : 
According to Bayesian theory, 

$$\begin{align}P(\lambda |D_1) & \propto P(\lambda) P(D_1|\lambda) \\
&=\frac{{b_0}^{a_0}}{\Gamma(a_0)} \lambda^{a_0-1} e^{-b_0\lambda} \prod_{i=1}^{n}P(d_i|\lambda) \\
&= \frac{{b_0}^{a_0}}{\Gamma(a_0)}\lambda^{a_0-1} e^{-b_0\lambda} \prod_{i=1}^{n}\frac{\lambda^{d_i} e^{-\lambda}}{d_i!} \\
&\propto \lambda^{a_0-1} e^{-b_0\lambda}\prod_{i=1}^{n}\lambda^{d_i} e^{-\lambda}\\
&= \lambda^{a_0+ \sum_{i=1}^{n} d_i-1} e^{-(b_0 +n)\lambda} \\
&= \text{Gamma}(a_n, b_n)
\end{align}$$


- larger sample size will lead to more concentrated posterior distribution; which can be seen from the variance of a Gamma distribution: $\text{Var}[\lambda|D]=a_n/b_n^2$, larger $n$ will leads to smaller variance; btw the mean of a Gamma is $a_n/b_n$

- ``a_0, b_0`` are called pseudo observations encapsulated in the prior; due to the previous mean and variance interpretation of a Gamma, $b_0$ represents how many pseudo observations in your prior distribution, and $a_0$ represents the sum of the pseudo observations; together they shape your prior belief about the unknown parameter $\lambda$.
"""

# ╔═╡ c900698d-577d-4d42-bee5-823724771d38
begin
	function showQ1(a0, b0, trueλ)
		xx = collect(0.1:0.1:15)
		ns = [5, 10, 20, 50, 100, 300]
		plt = plot(xx, pdf(Gamma(a0, 1/b0), xx), label="λ0")
		for n_i in ns
			D_i = rand(Poisson(trueλ), n_i)
			a_n = a0 + sum(D_i)
			b_n = b0 + n_i
			plot!(plt, xx, pdf(Gamma(a_n, 1/b_n), xx), label=string("λ",n_i))
		end
		return plt
	end
	showQ1(1, 1, 10)
end

# ╔═╡ d6a367e4-3a76-4a21-85ee-36cafc2a4749
begin
	a0 = 1. ; b0 = 1.;
	n = length(D1)
	an = a0 + sum(D1);
	bn = b0 + n
	mc = 1000
	λs = rand(Gamma(an, 1/bn), mc)
	# p1=plot(λs)
	# h1=histogram(λs)
	# plot(p1, h1)
	plot(Chains(λs))
end

# ╔═╡ 493d8a65-744d-4ea2-80d1-01e429643916
md"""
## Question 2 (Missing data handling)
You are given data set $D_2$, a sample of another n=100 counting data (i.e. $d_i \in \{0,1,2,\ldots\}$), assumed Poisson distributed again. To put the problem into some real world perspective, let's assume $d_i$ is blood count measurements of some patient. However, the sensor that made the observations is not reliable: some observations are missing at random. Make Bayesian inference over the mean $\lambda$ of the Poisson. 
!!! warning "... more advanced question to think about"
    what if the observation is multivariate Gaussian?
"""

# ╔═╡ 2aabe7b7-6381-4e78-b0d2-a25aa7d55d68
begin
	n2 = 100
	trueλ2 = 150.
	Dtmp = rand(Poisson(trueλ2), n2)
	D2 = Vector{Union{Int,Missing}}(undef, n2)
	missRate = 0.5
	oidx = shuffle(1:n2)[1:Int(floor((1-missRate)* n2))]
	D2[oidx] = Dtmp[oidx]
	plot(D2)
end

# ╔═╡ 91c3b329-68b8-411c-bfcf-3bd7715050fd
md"""
## Question 3 (Change point detection)
You are given data set $D_3=[d_1, d_2, \ldots, d_T]$, a time series of T blood count observations. Let's assume the data is blood count measurements of some patient over time. The patient has taken some treatment at some unknown point $t_0 \in [1, T)$. Assume his/her blood count changes significantly before and after the treament, which implies you should model the two period's blood counts $D_{30}=\{d_1, \ldots, d_{t_0 -1}\}$ and $D_{31}=\{d_{t_0}, \ldots, d_T\}$ as two Poissons, $\lambda_0$, $\lambda_1$. When did he take the treament, and what is the change of the blood count? 

!!! hint 
	what is the generative probabilistic model here ? what are the unknown or hidden random variables ?

"""

# ╔═╡ bf3d0a52-14ca-40c3-bc36-f8df7b7b9c19
begin
	T = 100
	thred = 20
	t0 = rand(thred:(T-thred))
	λ30 = 100
	λ31 = 150
	D3 = Vector{Int}(undef, T)
	D3[1:(t0-1)] = rand(Poisson(λ30), t0-1)
	D3[t0:end] = rand(Poisson(λ31), T-t0+1)
	plot(D3)
end

# ╔═╡ 79fc985a-c3fa-485b-b4a4-f3d7ed64fcd4
md"""
## Question 3b (Continuous change point)

Now let's consider a more interesting problem where the change point $t_0$ is no longer discrete but continuous. Suppose you are given a regression dataset $D_{3b}$, the dataset looks like below. Clearly, one linear regression model does not fit well as the regression line changes at some unknown point $x_0$. What is the change point and what is the best fit regression models?


!!! hint 
	You need to use Metropolis-Hasting sampling to sample the non-standard distribution of $t_0$.

"""

# ╔═╡ aaf1a19a-9e92-46d8-bafe-5489f8f66eeb
begin
	n2b = 100
	ncovar = 1
	D3b = zeros(n2b, ncovar+2)
	xmin = -10
	xmax = 10
	thredb = 2.5
	D3b[:,1] .= 1.
	D3b[:, 2:(ncovar+1)] = rand(Uniform(xmin,xmax),n2b,ncovar)
	t0b = rand(Uniform(xmin+thredb, xmax-thredb), ncovar)
	β0 = 2*rand(MvNormal(zeros(ncovar+1), 1.0 *Matrix(I, ncovar+1, ncovar+1)))
	β1 = zeros(ncovar+1)
	β1[2:end] = -1. * β0[2:end] 
	t0_ = [1.]
	append!(t0_, t0b)
	β1[1] = β0' * t0_ - β1[2:end]' * t0b
	stes = [2.0, 2.0]
	for i in 1:n2b 
		if D3b[i,2] < t0b[1]
			y_i = rand(Normal(D3b[i,1:(end-1)]' * β0, stes[1]))
		else
			y_i = rand(Normal(D3b[i,1:(end-1)]' * β1, stes[2]))
		end
		D3b[i,end] = y_i
	end
	scatter(D3b[:,2], D3b[:,3], xlabel="x", ylabel="y")
end

# ╔═╡ 20c56175-ca82-4d87-a10d-2930eb644024
md"""
## Question 4 (Truncated observation)
You are given data set $D_4=[d_1, d_2, \ldots, d_N]$, a sample of N blood count observations. However, the sensor that made the observation is not sensitive when the real count is small. It means there is unknown threshold $h > 0$ such that all observations $d < h$ is not reported. What is this threshold $h$ and what is the true blood count?

!!! hint
	The gibbs sampling steps here no-longer have closed-form distributions. You might need to use Metrapolis Hasting sampling here within the Gibbs iterations.
"""

# ╔═╡ 045fc907-9ce1-4c5a-a291-8d741b619563
begin
	h = 70
	λ4 = 60
	Nn = 300
	D4_ = rand(Poisson(λ4), Nn)
	D4 = D4_[D4_ .>= h]
	N4 = length(D4)
	D4b = Vector{Union{Missing, Int64}}(undef, Nn)
	D4b[D4_ .>= h] = D4_[D4_ .>= h]
	histogram(D4, normed=true, label="truncated")
	histogram!(D4_, normed= true, label ="real")
end

# ╔═╡ 2e368ab7-c37a-4ee0-aacb-554686a5d8a6
md"""
## Question 4b (Truncated with missing logs)
You are given data set $D_{4b}$, a sample of N blood count observations. The sensor that made the observation is not sensitive enough to distinguish when the real siginal is small, say $d < h$. The sensor logs the observation as missing whenever it happens. What is the sensor's detection range $h$ and what is the true blood count?

!!! warning "Can missing data be informative ?"
	You may think this question is the same as Q4. The two problems look similar but quite different, one is under-reporting while the other still reports "missing". For example, assume the sensor was perfect, and it has observed a dataset, say $[1, 2, 3, 7, 9]$. If $h=4$, Q4's dataset would be $[7,9]$ while this question's is $[\text{missing}, \text{missing}, \text{missing}, 7, 9]$. So the question boils down to whether the forms of missing information can be informative ? 

- can you think of a situation that Q4b's inference results are more accurate?
- under which situation Q4 and Q4b's difference is marginal ?
"""

# ╔═╡ Cell order:
# ╠═71d93c0d-0e4a-4684-8a54-6e25c2606a2b
# ╟─fef01c13-fc9e-45f7-9959-0a740f27b458
# ╟─797629b6-0669-11ec-3778-d1f20c3d6a4c
# ╟─4570ae10-8419-4af4-aea6-c5fc05756617
# ╠═13cb28fb-6b72-4ccf-9a2e-aec107ccd04b
# ╟─92d4bbd4-a7c1-426a-8c26-a40a4cb37cff
# ╟─c900698d-577d-4d42-bee5-823724771d38
# ╠═d6a367e4-3a76-4a21-85ee-36cafc2a4749
# ╟─493d8a65-744d-4ea2-80d1-01e429643916
# ╠═2aabe7b7-6381-4e78-b0d2-a25aa7d55d68
# ╟─91c3b329-68b8-411c-bfcf-3bd7715050fd
# ╟─bf3d0a52-14ca-40c3-bc36-f8df7b7b9c19
# ╟─79fc985a-c3fa-485b-b4a4-f3d7ed64fcd4
# ╟─aaf1a19a-9e92-46d8-bafe-5489f8f66eeb
# ╟─20c56175-ca82-4d87-a10d-2930eb644024
# ╟─045fc907-9ce1-4c5a-a291-8d741b619563
# ╟─2e368ab7-c37a-4ee0-aacb-554686a5d8a6
