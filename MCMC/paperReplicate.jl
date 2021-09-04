### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 57efbe58-090f-11ec-32d5-f336855f0481
begin
	cd(@__DIR__)
	using Pkg
	Pkg.activate(".")
	# Pkg.add("KernelDensity")
	# Pkg.add("FiniteDifferences")
	# Pkg.add("ForwardDiff")
	# Pkg.add("Turing")
	Pkg.add("LazyArrays")
	using Distributions, Random, MCMCChains, Plots
	plotly()
	using KernelDensity
	import StatsFuns:logsumexp as logsumexp
	using StatsPlots
	using LinearAlgebra
	using StatsBase
	using StatsFuns:logistic as logistic
	using PlutoUI
	using BenchmarkTools
	using FiniteDifferences
	using ForwardDiff
	using Turing
	using LazyArrays
end

# ╔═╡ 9bb78d8b-37d2-4fd7-8165-e3f40d6da45e
md"""
## Question 1: how to estimate $\pi$ by Monte Carlo method ?
"""

# ╔═╡ 31f2002b-1961-4646-b1c0-1b6453fefd51
begin
	function estimateπ(nn)
		spls = rand(Uniform(0,1), nn, 2)
		πfun(x) = Int(sqrt(norm(x)) < 1)
		return spls, 4. * mcestimate(spls, πfun)
	end
	
	function mcestimate(samples, func)
		est = mapslices(func, samples; dims= 2)
		return mean(est)
	end
	
	samples, _ = estimateπ(1000)
	
	scatter(samples[:,1], samples[:,2], markersize = 1.5, label="", ratio=1, xlim= (-0.01, 1.01), ylim= (- 0.01, 1.01))
	plot!([0, 0, 0, 1, 1, 1, 0, 1], [0, 1, 0, 0, 0, 1, 1, 1], linewidth= 2, linecolor = :black, label="")
	xvalues = collect(0:0.01:1)
	plot!(xvalues, sqrt.(1 .- xvalues.^2), linewidth= 2, linecolor = :red, label="")
	# π0 = 4* sum(sum(samples.^2, dims=2) .< 1)/nn
end

# ╔═╡ b3042d59-2b22-40f6-9f44-3a092d2d52b6
md""" Assume $P(\boldsymbol{x}) = \begin{cases} 1 & \boldsymbol{x} \in [0,1]^2\\ 0 & \text{otherwise}\end{cases}$; then we have 

$$\begin{align}\frac{1}{4} \pi R^2 = P(\sqrt{\boldsymbol{x}^T\boldsymbol{x}} < 1) &= \int I\left (\sqrt{\boldsymbol{x}^T\boldsymbol{x}} <1 \right) p(\boldsymbol{x}) d\boldsymbol{x} \\ &= E\left[I\left(\sqrt{\boldsymbol{x}^T\boldsymbol{x}} <1 \right)\right] \\
&\approx \frac{1}{m} \sum_{i=1}^m I\left(\sqrt{(\boldsymbol{x}^{(i)})^T\boldsymbol{x}^{(i)}} <1\right) \end{align}$$

Therefore, $\hat{\pi} = \frac{4}{m} \sum_{i=1}^m I(\sqrt{(\boldsymbol{x}^{(i)})^T\boldsymbol{x}^{(i)}} <1)$ """

# ╔═╡ 0848f104-2a35-424f-8706-1fc5ab010481
begin
	nsize = [50, 100, 5000]
	ntimes = 10
	πs = zeros(ntimes, length(nsize))
	for n_ in 1:length(nsize)
		for i in 1:ntimes
			_, πs[i, n_] = estimateπ(nsize[n_])
		end
	end
end

# ╔═╡ 63f5918c-fa02-4abf-a657-807506593f14
mean(πs, dims=1)

# ╔═╡ 9369c5db-79a0-4667-b6f3-583d720dee72
begin
	function posterior1(x; logP=true)
		p = logsumexp([log(0.3)-0.2*x^2, log(0.7) - 0.2*(x-10)^2])
		logP ? p : exp(p)
	end
	
	# function posterior0(x)
	# 	# logp = logsumexp([log(0.3) + (-0.2*x^2), log(0.7) - 0.2*(x-10)^2])
	# 	0.3*exp(-0.2*x^2) + 0.7*exp(-0.2*(x-10)^2)
	# end
	
	function MH(pf; logP = true, σ2=100, x0=0., mc=5000)
		samples = zeros(mc)
		σ = sqrt(σ2)
		q = Normal(x0, σ)
		accR = 0;
		pfx0 = pf(x0)
		for i in 1:mc
			xstar = rand(q)
			pfxstar = pf(xstar)
			if logP
				Acp = pfxstar + logpdf(q, x0) - pfx0 - logpdf(q, xstar)
				Acp = exp(Acp)
			else
				Acp = pfxstar * pdf(q, x0) / (pfx0 * pdf(q, xstar))
			end
			if rand() < Acp
				x0 = xstar
				pfx0 = pfxstar
				accR += 1
			end
			samples[i] = x0
		end
		return samples, accR/mc
	end
	
	
	function MHRW(pf; logP = true, σ2=100, x0=0., mc=5000)
		samples = zeros(mc)
		σ = sqrt(σ2)
		q = Normal(x0, σ)
		accR = 0;
		pfx0 = pf(x0) 
		for i in 1:mc
			xstar = rand(q)
			qstar = Normal(xstar, σ)
			q = Normal(x0, σ)
			pfxstar = pf(xstar)
			if logP
				Acp = pfxstar + logpdf(qstar, x0) - pfx0 - logpdf(q, xstar)
				Acp = exp(Acp)
			else
				Acp = pfxstar * pdf(q, x0) / (pfx0 * pdf(q, xstar))
			end
			if rand() < Acp
				x0 = xstar
				pfx0 = pfxstar
				accR += 1
			end
			samples[i] = x0
		end
		return samples, accR/mc
	end
	mcsample, ar100 = MH(posterior1; σ2= 100, mc=5000);
	mcsampl100Chain = Chains(mcsample);
	with_terminal() do
		print("acceptance rate for σ2 = 100 is: ", ar100)
	end
end

# ╔═╡ 9d583ae3-1396-4529-b9c2-0dc4edeaa4b8
begin
	xx = collect(-10:0.1:20)
	postxx = posterior1.(xx; logP=false)
	p1=plot(xx, postxx/1.5, linewidth=2, label="I=100")
	histogram!(p1, mcsample[1:100], bins = 80, normalize=true, label="")
	p2=plot(xx, postxx/1.5, linewidth=2, label="I=500")
	histogram!(p2, mcsample[1:500], bins = 100, normalize=true, label="")
	
	p3=plot(xx, postxx/1.5, linewidth=2, label="I=1000")
	histogram!(p3, mcsample[1:1000], bins = 100, normalize=true, label="")

	p4=plot(xx, postxx/1.5, linewidth=2, label="I=5000")
	histogram!(p4, mcsample[1:5000], bins = 100, normalize=true, label="")
	plot(p1, p2, p3, p4)
end

# ╔═╡ d21773b6-fd4a-4218-98c0-150f53940348
begin
	mcsample100RW, arRW100 = MHRW(posterior1; σ2= 100, mc=5000);
	mcsampl100RWChain = Chains(mcsample100RW);
	with_terminal() do
		print("acceptance rate for MHRW σ2 = 100 is: ", arRW100)
	end
	p1rw=plot(xx, postxx/1.5, linewidth=2, label="MHRW I=100")
	histogram!(p1rw, mcsample100RW[1:100], bins = 80, normalize=true, label="")
	p2rw=plot(xx, postxx/1.5, linewidth=2, label="MHRW I=500")
	histogram!(p2rw, mcsample100RW[1:500], bins = 100, normalize=true, label="")
	
	p3rw=plot(xx, postxx/1.5, linewidth=2, label="MHRW I=1000")
	histogram!(p3rw, mcsample100RW[1:1000], bins = 100, normalize=true, label="")

	p4rw=plot(xx, postxx/1.5, linewidth=2, label="MHRW I=5000")
	histogram!(p4rw, mcsample100RW[1:5000], bins = 150, normalize=true, label="")
	plot(p1rw, p2rw, p3rw, p4rw)
end

# ╔═╡ 6334edb5-c5b0-4868-8398-56c00646148a
begin
	plot(mcsampl100RWChain)
end

# ╔═╡ 29a3de2d-7706-4ea5-ada7-5da6d29c4f3e
begin
	d1 = density(vec(mcsample[1:100]), label="I=100")
	density!(vec(mcsample[1:500]), label="I=500")
	# density!(vec(mcsample[1:1000]), label= "I=250")
	density!(vec(mcsample[1:5000]), label="I=5000")
	plot!(xx, postxx./4.35, linewidth=2, label="true")
end

# ╔═╡ 275f72c2-ac3b-4d25-ae1f-931a7b0801c8
plot(mcsampl100Chain)

# ╔═╡ 608b70f9-8f7e-4dcd-a4ec-87358e7383ae
begin
	mcsample1, ar1 = MH(posterior1; σ2= 1, mc=5000);
	mcsampl1Chain = Chains(mcsample1)
	mcsample10000, ar10000 = MH(posterior1; σ2= 100^2, mc=5000);
	mcsampl10000Chain = Chains(mcsample10000);
	with_terminal() do
		println("acceptance rate for σ2 = 1 is: ", ar1)
		println("acceptance rate for σ2 = 100 is: ", ar100)
		println("acceptance rate for MHRW σ2 = 100 is: ", arRW100)
		println("acceptance rate for σ2 = 10000 is: ", ar10000)
		println("Is acceptance rate higher the better ? ")
	end
	# plot(mcsampl1Chain)
end

# ╔═╡ aa314f7d-23c2-4dcb-97cd-945fe8995e55
plot(mcsampl10000Chain)

# ╔═╡ dbfa7c16-0093-4131-b3e2-e76ade890fa5
summarize(mcsampl1Chain)

# ╔═╡ a5116149-f84c-4fb3-9b4f-1826865c4d62
summarize(mcsampl100Chain)

# ╔═╡ 6fc271ae-ecd8-4773-8787-be3eca2456af
summarize(mcsampl100RWChain)

# ╔═╡ 472ffbe6-f6a5-4567-8e92-a30b87c3e5f6
summarize(mcsampl10000Chain)

# ╔═╡ 384ed78e-8cd3-410d-a167-7c3b4f09a282
function MH_SA(logP; σ2=100, x0=0., mc=5000, C=1, T0=1)
	T = T0
	samples = zeros(mc)
	σ = sqrt(σ2)
	q = Normal(x0, σ)
	for i in 1:mc
		xstar = rand(q)
		qstar = Normal(xstar, σ)
		Acp = (1/T)*logP(xstar) + logpdf(q, x0) - (1/T)*logP(x0) - logpdf(q, xstar)
		if rand() < exp(Acp)
			x0 = xstar
		end	
		samples[i] = x0
		T = (C*log(i+T0))^(-1)
	end
	return samples
end

# ╔═╡ 8515c7fa-4509-48a5-ab27-09894ec696ce
begin
	splSA = MH_SA(posterior1; σ2= 100, mc=5000, C=10);
end

# ╔═╡ 1bd57c8f-e169-4359-9f51-cc733ce0bf19
begin
	p1_=plot(xx, postxx/1.5, linewidth=2, label="I=100")
	histogram!(p1_, splSA[1:100], bins = 80, normalize=true, label="")
	p2_=plot(xx, postxx/1.5, linewidth=2, label="I=500")
	histogram!(p2_, splSA[1:500], bins = 100, normalize=true, label="")
	
	p3_=plot(xx, postxx/1.5, linewidth=2, label="I=1000")
	histogram!(p3_, splSA[1:1000], bins = 100, normalize=true, label="")

	p4_=plot(xx, postxx/1.5, linewidth=2, label="I=5000")
	histogram!(p4_, splSA[1:5000], bins = 100, normalize=true, label="")
	plot(p1_, p2_, p3_, p4_)
end

# ╔═╡ 7b1cad31-5d80-4b18-a66e-6f69babf6f41
plot(splSA)

# ╔═╡ b85f10cb-3a94-4cb8-b7e0-d00d0a957e6a
begin
	
	function create_interval(x, u, pf, w, max_steps)
		L = x - rand(Uniform(0, w))
		R = L +w
		
# 		step out
		J = floor(max_steps * rand())
		K = (max_steps - 1) - J
		while u < exp(pf(L)) && J > 0
			L = L - w
			J = J - 1
		end
		
		while u < exp(pf(R)) && K >0
			R = R +w
			K = K -1
		end
		
		return L, R
	end
	
	
	function shrink_and_sample(x, u, pf, interval)
		L = interval[1]
		R = interval[2]
		
		while true
			x0 = rand(Uniform(L, R))
			if u < exp(pf(x0))
				return x0
			end
			
# 			shrink
			if x0 > x
				R = x0
			end
			
			if x0 < x
				L = x0
			end
		end
		
	end
	
	function slice(pf; x0, mc, w, max_steps)
		us = zeros(mc) 
		samples = zeros(mc)
		
		for i in 1:mc
			us[i] = u0 = rand(Uniform(0, exp(pf(x0))))
			
			x0 = shrink_and_sample(x0, u0, pf, create_interval(x0, u0, pf, w, max_steps))
			samples[i] = x0
		end
		
		return samples, us
	end
		
end

# ╔═╡ bd55f18c-fbf2-47d7-8936-f6fb54f9b45f
begin
	slsamples, uslsamples = slice(posterior1; x0 = 0., mc= 5000, w= 5, max_steps= 10000);
	slsamplesChain = Chains(slsamples)
	summarize(slsamplesChain)
end

# ╔═╡ b43f0f7d-73df-4f85-a877-d2cb9c89c801
histogram(slsamples, nbin=100)

# ╔═╡ 1b5c1184-ec24-41d2-8e8f-662bc72cdb62
scatter(slsamples, uslsamples, m=:auto, markersize = 1)

# ╔═╡ 9ff130c3-8c0d-4ce9-9908-ac836f97aadb
# begin
# 	x0 = 0.
# 	u0 = rand(Uniform(0, exp(posterior1(x0))))
# 	L0, R0 =create_interval(x0, u0, posterior1, 5, 10000)
# 	plot(xx, postxx)
# 	plot!([L0, R0], [u0, u0])
# 	intv0 = [L0, R0]
# 	x1 = shrink_and_sample(x0, u0, posterior1, intv0);
# 	scatter!([x0], [u0], m =:cross, markersize=5, label="x0")
# 	scatter!([x1], [u0], m =:diamond, markersize=5, label="x1")
# end

# ╔═╡ 9881e098-4b9a-402d-896e-74b5b1934da1
md"""
### Importance sampling
"""

# ╔═╡ 62769ead-22b4-4856-8f49-36d70943c018
begin
	wμ = 7
	wσ2 = 100
	wmc = 1000
	wdensity = Normal(wμ, sqrt(wσ2))
	# wdensity = MixtureModel(Normal[Normal(0, sqrt(5/2)), Normal(10, sqrt(5/2))], [0.3, 0.7])
	wsamples = rand(wdensity, wmc);
	wtildes = (posterior1.(wsamples) - logpdf(wdensity, wsamples))
	ws = exp.(wtildes .-logsumexp(wtildes))
end

# ╔═╡ 209089a3-81f7-438c-9770-a0d51833252c
ws' * wsamples

# ╔═╡ a13e743b-22c5-47df-8a51-6debc605c7f7
sum(ws.^2)^(-1)

# ╔═╡ c9ab23b2-2940-42dd-82a9-8027482ed046
begin
	resampleSize = 1000
	resplIDs = sample(1:length(ws), Weights(ws), resampleSize)
	v0 = pdf(Normal(wμ, sqrt(wσ2)), wμ)
	max0 = posterior1(10; logP=false)
	h = fit(Histogram, wsamples[resplIDs], nbins=100)
	h = normalize(h, mode=:pdf)
	h.weights = h.weights / maximum(h.weights)
	plot(h, label="resampled samples", title="Sampling importance resampling")
	plot!(xx, pdf(Normal(wμ, sqrt(wσ2)), xx)/pdf(Normal(wμ, sqrt(wσ2)), wμ), linecolor=:red, linewidth=2, label="importance Density")
	plot!(xx, postxx/posterior1(10; logP=false), linecolor=:black, linewidth=2, label="true density")
end

# ╔═╡ 57461a66-c738-41da-9b8e-2882ae1b65fb
md"""
### Bayesian Logistic Regression


What is the statistical model for a logistic regression? Bayesian model ? 
"""

# ╔═╡ 00a16917-1e41-4137-9108-aa284b423db0
md"""

Check the scatter plot below
- How would you label the diamond point below? 
- How would you draw a decision boundary that best reflects the situation ?
"""

# ╔═╡ 86068a8b-c169-4da9-9011-ef6bc5477c43
begin
	# mm = 50
	dim = 2
# 	w0 = rand(Normal(), dim)
# 	XX = rand(Uniform(-1, 1), mm, dim-1)
# 	XX = hcat([ones(mm), XX]...)
# 	yy = Int64.(rand.(Bernoulli.(logistic.(XX * w0))))
	
# 	XXtest = rand(Uniform(-1, 1), mm, dim-1)
# 	XXtest = hcat([ones(mm), XXtest]...)
# 	yytest = Int64.(rand.(Bernoulli.(logistic.(XXtest * w0))))
	n0 = 5
	n1 = 5
	μ1 = 3.
	XX1 = rand(MvNormal(μ1 .*ones(2), 1.5 .*Matrix(I,2,2)), n1)
	XX0 = rand(MvNormal(-μ1 .*ones(2), 1.5 .*Matrix(I,2,2)), n0)
	XX = [XX0'; XX1']
	yy = [zeros(n0); ones(n1)]
	
	
	XX1_ = rand(MvNormal(μ1 .*ones(2), 5 .*Matrix(I,2,2)), n1)
	XX0_ = rand(MvNormal(-μ1 .*ones(2), 5 .*Matrix(I,2,2)), n0)
	XXtest = [XX0_'; XX1_']
	yytest = [zeros(n0); ones(n1)]
	scatter(XX0[1,:], XX0[2, :], label="0")
	scatter!(XX1[1,:], XX1[2, :], label="1")
	scatter!([-4], [3], markershape= :diamond, label="?")
end

# ╔═╡ 7f0712c5-ef93-4120-b1e3-625a16107bca
begin
	
	function posteriorLR(w; m0, V0, X, y)
		σ = logistic.(X * w)
		Λ0 = inv(V0)
		grad = - inv(V0) * (w-m0) + X' * (y - σ)
		d = σ .* (σ .- 1)
		H = (X .* d)' * X - Λ0
		return logpdf(MvNormal(m0, V0), w) + sum(logpdf.(Bernoulli.(σ), y)), grad, H
	end

	function posteriorLR(w0, w1; m0, V0, X, y)
		w = [w0, w1]
		logpost, _ , _ = posteriorLR(w; m0=m0, V0=V0, X=X, y=y)
		return logpost
	end
end

# ╔═╡ fb901676-0cc4-42e1-a38e-7d2a3d0d86ee
begin
	x00 = rand(dim)
	rstfd = grad(central_fdm(5,1), (x) -> posteriorLR(x; m0=zeros(dim), V0=100 .* Matrix(I, dim, dim), X=XX, y=yy)[1], x00)[1]
end

# ╔═╡ 8adf4b04-85a2-46fc-9814-0d213e80b57c
begin	
	
function MHRWMvN(pf, dim; logP = true, Σ = 10. *Matrix(I, dim, dim), x0=zeros(dim), mc=5000, burnin =0, thinning = 1)
		samples = zeros(mc, dim)
		C = cholesky(Σ)
		L = C.L
		# q = MvNormal(x0, Σ)
		pfx0 = pf(x0) 
		j = 1
		for i in 1:((mc+burnin)*thinning)
			# q = MvNormal(x0, Σ)
			xstar = x0 + L * randn(dim)
			pfxstar = pf(xstar)
			if logP
				Acp = pfxstar - pfx0 
				Acp = exp(Acp)
			else
				Acp = pfxstar / pfx0 
			end
			if rand() < Acp
				x0 = xstar
				pfx0 = pfxstar
			end
			if i > burnin && mod(i, thinning) ==0
				samples[j,:] = x0
				j += 1
			end
		end
		return samples
	end
	
end

# ╔═╡ 36381da3-8709-48f3-818c-51a355a92c37
begin
	
	function MCMCLogisticR(X, y, dim; mc= 1000, burnin=0, thinning=10, m0= zeros(dim), V0 = 100 .* Matrix(I, dim, dim))
		postLRFun(x) = posteriorLR(x; m0=m0, V0=V0, X=X, y=y)
		wt, Vt, _ = LaplaceLogisticR(X, y, dim; m0 = m0, V0=V0)
		qV = (2.38^2/dim)*Vt
		mcLR = MHRWMvN((x) -> postLRFun(x)[1], dim; logP = true, Σ = qV, x0=wt, mc=mc, burnin=burnin, thinning= thinning)
		return mcLR
		# return wt, Ht
	end
	
	function LaplaceLogisticR(X, y, dim; m0= zeros(dim), V0 = 100 .* Matrix(I, dim, dim), maxIters = 1000, tol= 1e-4)
		wt = zeros(dim)
		fts = zeros(maxIters)
		Ht = zeros(dim, dim)
		postLRFun(x) = posteriorLR(x; m0=m0, V0=V0, X=X, y=y)
		for t in 1:maxIters
			fts[t], gt, Ht = postLRFun(wt)
			wt = wt - Ht\gt
			if t > 1 && abs(fts[t] - fts[t-1]) < tol
				fts = fts[1:t]
				break
			end
		end
		V = Hermitian(inv(-1*Ht))
		return wt, V, fts
	end
end

# ╔═╡ b99dba21-c185-4bb0-98e3-b76ef790aca8
mLP, VLP, _ = LaplaceLogisticR(XX, yy, dim; V0 = 1e3 * Matrix(I, dim, dim));

# ╔═╡ 5e951aa4-6d0f-4e44-a972-d078e5e32810
mcLR = MCMCLogisticR(XX, yy, dim; mc=2000, V0 = 1e3 * Matrix(I, dim, dim));

# ╔═╡ f3191136-1739-4734-ad54-e66aac3cc2db
describe(Chains(mcLR))

# ╔═╡ 5e6f5bf0-0749-4c39-922a-e4c4590e1074
plot(Chains(mcLR))

# ╔═╡ fde63878-258b-4001-ad4c-d78958074459
corner(Chains(mcLR))

# ╔═╡ 5e1fe060-c5b1-491f-b76a-8304270d2649
begin
	pflogLR(w1, w2) = posteriorLR(w1, w2; m0=zeros(dim), V0 = 1e3 * Matrix(I, dim, dim), X=XX, y=yy)
	w1 = -50:1:120
	w2 = -60:1:150
	contour(w1, w2, pflogLR, xlabel="w1", ylabel="w2", fill=true,  connectgaps=true, line_smoothing=0.85, title="")
end

# ╔═╡ 261feb07-8310-4de0-80a9-4a4fd0c58754
plot(w1,w2,pflogLR,st=:surface)

# ╔═╡ 97e6a5d5-fa9b-4af8-a8a2-a8013d9e7bd2
begin
	mcLP = rand(MvNormal(mLP, Matrix(VLP)), 2000)'
	corner(Chains(mcLP))
end

# ╔═╡ 57a82871-0f0d-4714-9e54-f60d99b9ec61
@model logreg(X,  y; predictors=size(X, 2)) = begin
    #priors
    # α ~ Normal(0, sqrt(100))
    β ~ filldist(Normal(0, sqrt(1000)), predictors)

    #likelihood
    y ~ arraydist(LazyArray(@~ BernoulliLogit.(X * β)))
end;

# ╔═╡ 6df7c521-9993-47c0-abe9-a0422903eea2
model = logreg(XX, yy);

# ╔═╡ 965d864c-8efe-4d6f-baa9-e479926d3abe
begin
	chain = sample(model, NUTS(),  2000)
	describe(chain)
	# corner(chain)
end

# ╔═╡ f9d8fe2e-09da-4af3-abee-a5b9eace9e39
begin
	function predictiveLogLik(ws, X, y)
		mc = size(ws)[1]
		h = logistic.(X * ws')
		logLiks = sum(logpdf.(Bernoulli.(h), y), dims=1)
		log(1/mc) + logsumexp(logLiks)
	end
	
	
	function mcPrediction(ws, X)
		# mc = size(ws)[1]
		# h = logistic.(X * ws')
		mean(logistic.(X * ws'), dims=2)
	end
	
	function mcPrediction(ws, x1, x2)
		# mc = size(ws)[1]
		# h = logistic.(X * ws')
		mean(logistic.(ws * [x1, x2]))
	end
	
end

# ╔═╡ 0d238da9-915f-4f50-8239-e171cc9726ff
predictiveLogLik(mcLR, XXtest, yytest)

# ╔═╡ df1b05c6-91cf-48e1-a254-4f7b6654de69
predictiveLogLik(mcLP, XXtest, yytest)

# ╔═╡ 33c8fdc7-1965-4274-aa56-7ab573711ada
md"""
### Bayesian predictive distribution

$$P(y_{n+1}|x_{n+1}, D) = \int_{w}P(y_{n+1} |x_{n+1}, w)P(w|D) dw$$

Its Monte Carlo estimator is 

$$P(y_{n+1}|x_{n+1}, D) \approx \frac{1}{m} \sum_{i=1}^{m} P(y_{n+1}|x_{n+1}, w^{(i)})$$

1. What is $P(y_{n+1}|x_{n+1}, w^{(i)})$ ?

2. Comparing with frequentist's plugin point estimator, $P(y_{n+1}|x_{n+1}, w_{\text{MAP}})$, what is the difference ?

"""

# ╔═╡ b1f8d4ac-f0df-4999-9d46-eb8fe4f27f47
begin
# 	Frequentist MAP estimator prediction
	ppfLRPoint(x, y) = mcPrediction(mLP', x, y)
	x = -3:0.02:3
	y = -3:0.02:3
	contour(x, y, ppfLRPoint, xlabel="x", ylabel="y", fill=true,  connectgaps=true, line_smoothing=0.85, title="L2 MAP prediction", c=:roma)
end

# ╔═╡ ecabc51d-51a3-4b18-a9c3-507b624f3ebd
plot(x,y,ppfLRPoint,st=:surface, c=:roma)

# ╔═╡ 4c565f9d-4eed-475b-9613-32bfeeffcec9
begin
	ppf(x, y) = mcPrediction(mcLR, x, y)
	contour(x, y, ppf, xlabel="x", ylabel="y", fill=true,  connectgaps=true, line_smoothing=0.85, title="Bayesian MCMC prediction", c=:roma)
end

# ╔═╡ 0bfa1688-af3c-47f3-bf56-8a2b18828be4
plot(x,y,ppf,st=:surface, c=:roma)

# ╔═╡ 47d4a060-e027-4c94-a5fb-3af5b3873ff9
begin
# 	
	ppfLP(x, y) = mcPrediction(mcLP, x, y)
	# x = -1.:0.02:1.
	# y = -1.:0.02:1.
	contour(x, y, ppfLP, xlabel="x", ylabel="y", fill=true,   connectgaps=true, line_smoothing=0.85, title="Laplace Approximation Prediction", c=:roma)
end

# ╔═╡ 974f10e3-2347-4081-ba24-dc9ff414f832
plot(x,y,ppfLP,st=:surface, c=:roma)

# ╔═╡ Cell order:
# ╟─57efbe58-090f-11ec-32d5-f336855f0481
# ╟─9bb78d8b-37d2-4fd7-8165-e3f40d6da45e
# ╠═31f2002b-1961-4646-b1c0-1b6453fefd51
# ╟─b3042d59-2b22-40f6-9f44-3a092d2d52b6
# ╠═0848f104-2a35-424f-8706-1fc5ab010481
# ╠═63f5918c-fa02-4abf-a657-807506593f14
# ╠═9369c5db-79a0-4667-b6f3-583d720dee72
# ╠═9d583ae3-1396-4529-b9c2-0dc4edeaa4b8
# ╠═d21773b6-fd4a-4218-98c0-150f53940348
# ╠═6334edb5-c5b0-4868-8398-56c00646148a
# ╠═29a3de2d-7706-4ea5-ada7-5da6d29c4f3e
# ╠═275f72c2-ac3b-4d25-ae1f-931a7b0801c8
# ╠═608b70f9-8f7e-4dcd-a4ec-87358e7383ae
# ╠═aa314f7d-23c2-4dcb-97cd-945fe8995e55
# ╠═dbfa7c16-0093-4131-b3e2-e76ade890fa5
# ╠═a5116149-f84c-4fb3-9b4f-1826865c4d62
# ╠═6fc271ae-ecd8-4773-8787-be3eca2456af
# ╠═472ffbe6-f6a5-4567-8e92-a30b87c3e5f6
# ╠═384ed78e-8cd3-410d-a167-7c3b4f09a282
# ╠═8515c7fa-4509-48a5-ab27-09894ec696ce
# ╠═1bd57c8f-e169-4359-9f51-cc733ce0bf19
# ╠═7b1cad31-5d80-4b18-a66e-6f69babf6f41
# ╠═b85f10cb-3a94-4cb8-b7e0-d00d0a957e6a
# ╠═bd55f18c-fbf2-47d7-8936-f6fb54f9b45f
# ╠═b43f0f7d-73df-4f85-a877-d2cb9c89c801
# ╠═1b5c1184-ec24-41d2-8e8f-662bc72cdb62
# ╟─9ff130c3-8c0d-4ce9-9908-ac836f97aadb
# ╟─9881e098-4b9a-402d-896e-74b5b1934da1
# ╠═62769ead-22b4-4856-8f49-36d70943c018
# ╠═209089a3-81f7-438c-9770-a0d51833252c
# ╠═a13e743b-22c5-47df-8a51-6debc605c7f7
# ╠═c9ab23b2-2940-42dd-82a9-8027482ed046
# ╟─57461a66-c738-41da-9b8e-2882ae1b65fb
# ╟─00a16917-1e41-4137-9108-aa284b423db0
# ╠═86068a8b-c169-4da9-9011-ef6bc5477c43
# ╠═7f0712c5-ef93-4120-b1e3-625a16107bca
# ╠═fb901676-0cc4-42e1-a38e-7d2a3d0d86ee
# ╟─8adf4b04-85a2-46fc-9814-0d213e80b57c
# ╠═36381da3-8709-48f3-818c-51a355a92c37
# ╠═b99dba21-c185-4bb0-98e3-b76ef790aca8
# ╠═5e951aa4-6d0f-4e44-a972-d078e5e32810
# ╠═f3191136-1739-4734-ad54-e66aac3cc2db
# ╠═5e6f5bf0-0749-4c39-922a-e4c4590e1074
# ╠═fde63878-258b-4001-ad4c-d78958074459
# ╠═5e1fe060-c5b1-491f-b76a-8304270d2649
# ╠═261feb07-8310-4de0-80a9-4a4fd0c58754
# ╠═97e6a5d5-fa9b-4af8-a8a2-a8013d9e7bd2
# ╠═57a82871-0f0d-4714-9e54-f60d99b9ec61
# ╠═6df7c521-9993-47c0-abe9-a0422903eea2
# ╠═965d864c-8efe-4d6f-baa9-e479926d3abe
# ╟─f9d8fe2e-09da-4af3-abee-a5b9eace9e39
# ╠═0d238da9-915f-4f50-8239-e171cc9726ff
# ╠═df1b05c6-91cf-48e1-a254-4f7b6654de69
# ╟─33c8fdc7-1965-4274-aa56-7ab573711ada
# ╟─b1f8d4ac-f0df-4999-9d46-eb8fe4f27f47
# ╠═ecabc51d-51a3-4b18-a9c3-507b624f3ebd
# ╟─4c565f9d-4eed-475b-9613-32bfeeffcec9
# ╠═0bfa1688-af3c-47f3-bf56-8a2b18828be4
# ╟─47d4a060-e027-4c94-a5fb-3af5b3873ff9
# ╠═974f10e3-2347-4081-ba24-dc9ff414f832
