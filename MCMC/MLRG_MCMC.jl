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
end

# ╔═╡ 797629b6-0669-11ec-3778-d1f20c3d6a4c
md"""# MCMC: exercises
- Lei Fang 25/08/2021

Here are a few exercises you can try to improve your understanding of Bayesian inference and MCMC. As most of the problems listed below have no closed-form posterior distributions, you need to resort to MCMC sampler to approximate the true posteriors. You are recommended to do the exercises in the given order.
"""

# ╔═╡ 4570ae10-8419-4af4-aea6-c5fc05756617
md"""
### Question 1 (Conjugate inference)
You are given data set $D_1$, a sample of n=100 i.i.d (independently and identically distributed) observations from a Poisson distribution. A Poisson distribution has a probability mass function 

$$P(X=k|\lambda) = \frac{\lambda^k e^{-\lambda}}{k!},$$ where $\lambda >0$ is the mean of the distribution. Make a Bayesian inference over $\lambda$ given $D_1$. Hint: the conjugate prior for $\lambda$ is a Gamma distribution: 

$$P(\lambda) = \text{Gamma}(a_0,b_0) = \frac{{b_0}^{a_0}}{\Gamma(a_0)} \lambda^{a_0-1} e^{-b_0\lambda}.$$

Show the posterior is still a Gamma distribution with an updated parameters $a_n, b_n$: i.e.

$$P(\lambda|D) = \text{Gamma}(a_n, b_n),$$ where $a_n= a_0 + \sum_{i=1}^n d_i, b_n = b_0 +n.$ Then sample from the posterior.
"""

# ╔═╡ 13cb28fb-6b72-4ccf-9a2e-aec107ccd04b
begin
	trueλ1 = 10.
	n1 = 100
	D1 = rand(Poisson(trueλ1), n1)
end

# ╔═╡ 92d4bbd4-a7c1-426a-8c26-a40a4cb37cff
md"""**Solution** : 
"""

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

# ╔═╡ 20c56175-ca82-4d87-a10d-2930eb644024
md"""
## Question 4 (Truncated observation)
You are given data set $D_4=[d_1, d_2, \ldots, d_N]$, a sample of N blood count observations. However, the sensor that made the observation is not sensitive when the real count is small. It means there is unknown threshold $h > 0$ such that all observations $d < h$ is not reported. What is this threshold $h$ and what is the true blood count?
"""

# ╔═╡ 045fc907-9ce1-4c5a-a291-8d741b619563
begin
	h = 36
	λ4 = 40
	D4 = rand(Poisson(λ4), 100)
	D4 = D4[D4 .> h]
	N4 = length(D4)
	plot(D4)
end

# ╔═╡ Cell order:
# ╟─71d93c0d-0e4a-4684-8a54-6e25c2606a2b
# ╟─797629b6-0669-11ec-3778-d1f20c3d6a4c
# ╟─4570ae10-8419-4af4-aea6-c5fc05756617
# ╠═13cb28fb-6b72-4ccf-9a2e-aec107ccd04b
# ╠═92d4bbd4-a7c1-426a-8c26-a40a4cb37cff
# ╠═d6a367e4-3a76-4a21-85ee-36cafc2a4749
# ╟─493d8a65-744d-4ea2-80d1-01e429643916
# ╠═2aabe7b7-6381-4e78-b0d2-a25aa7d55d68
# ╟─91c3b329-68b8-411c-bfcf-3bd7715050fd
# ╠═bf3d0a52-14ca-40c3-bc36-f8df7b7b9c19
# ╟─20c56175-ca82-4d87-a10d-2930eb644024
# ╠═045fc907-9ce1-4c5a-a291-8d741b619563
