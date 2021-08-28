### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 71d93c0d-0e4a-4684-8a54-6e25c2606a2b
begin
	cd(@__DIR__)
	using Pkg
	Pkg.activate(".")
	# Pkg.add("StatsPlots")
	# Pkg.add("StatsFuns")
	# Pkg.add("SpecialFunctions")
	# Pkg.add("AdvancedMH")
	# Pkg.add("StructArrays")
	# Pkg.add("Optim")
	# Pkg.add("Profile")
	using Distributions, Random, MCMCChains, Plots
	plotly()
	using StatsPlots
	import StatsFuns:logsumexp as logsumexp
	import StatsFuns:logistic as logistic
	using SpecialFunctions
	using AdvancedMH
	using StructArrays
	using Optim
	using LinearAlgebra
	using Profile
end

# ╔═╡ 797629b6-0669-11ec-3778-d1f20c3d6a4c
md"""# MCMC: exercises
- Lei Fang 25/08/2021

Here are a few exercises you can try to improve your understanding of Bayesian inference and MCMC. As most of the problems listed below have no closed-form posterior distributions, you need to resort to MCMC sampler to approximate the true posteriors. You are recommended to do the exercises in the given order.
"""

# ╔═╡ 4570ae10-8419-4af4-aea6-c5fc05756617
md"""
### Question 1: Conjugate Inference
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
	D1 = rand(Poisson(trueλ1), n1);
end

# ╔═╡ 92d4bbd4-a7c1-426a-8c26-a40a4cb37cff
md"""**Solution** : 
"""

# ╔═╡ d6a367e4-3a76-4a21-85ee-36cafc2a4749
begin
	
	function sampleλ(D, a0=1., b0=1., mc=1000)
		n = length(D)
		an = a0 + sum(D);
		bn = b0 + n
		λs = rand(Gamma(an, 1/bn), mc)
		λs
	end
	λs1 = sampleλ(D1)
	# p1=plot(λs)
	# h1=histogram(λs)
	# plot(p1, h1)
	plot(Chains(λs1))
end

# ╔═╡ 493d8a65-744d-4ea2-80d1-01e429643916
md"""
## Question 2: Missing Data
You are given data set $D_2$, a sample of another n=100 counting data (i.e. $d_i \in \{0,1,2,\ldots\}$), assumed Poisson distributed again. To put the problem into some real world perspective, let's assume $d_i$ is blood count measurements of some patient. However, the sensor that made the observations is not reliable: some observations are missing at random. Make Bayesian inference over the mean $\lambda$ of the Poisson. 
"""

# ╔═╡ 2aabe7b7-6381-4e78-b0d2-a25aa7d55d68
begin
	n2 = 10
	trueλ2 = 2.
	Dtmp = rand(Poisson(trueλ2), n2)
	D2 = Vector{Union{Int,Missing}}(undef, n2)
	missRate = 0.5
	oidx = shuffle(1:n2)[1:Int(floor((1-missRate)* n2))]
	D2[oidx] = Dtmp[oidx]
	plot(D2)
end

# ╔═╡ c7d2d072-1551-4901-8cb6-1b222d666b93
md"""
**Solution:**
"""

# ╔═╡ 71a5896b-2e3b-4ba4-8ded-9b46e5ddcbd6
begin
	function gibbsMissingPoisson(D, a0=1., b0=1., mc=1000, burnin =1000)
		missingIdx = ismissing.(D)
		Dworking = copy(D)
		λt = mean(D[.!missingIdx])
		bn = b0 + length(D)
		λs = zeros(mc)
		for i in 1:(mc+burnin)
	# 		sample missing data
			Dworking[missingIdx] = rand(Poisson(λt), sum(missingIdx))
	# 		sample λ
			an = a0 + sum(Dworking)
			λt = rand(Gamma(an, 1/bn))
			if i >burnin
				λs[i-burnin] = λt
			end
		end
		return λs
	end
	λmc = gibbsMissingPoisson(D2, 0.1, 0.1, 5000, 1000);	
	# plot(Chains(λmc[2:end]))
	
	# an_ = 0.1 + sum(D2[oidx])
	# bn_ = 0.1 + length(oidx)
	# λmc2 = rand(Gamma(an_, 1/bn_), 5000);
	λmc2 = sampleλ(D2[oidx], 1, 1, 5000)
	λs2= hcat([λmc, λmc2]...)
	plot(Chains(λs2))
end

# ╔═╡ 369df433-38b1-4dbf-982c-2799302b7d8f
begin
	function simulateMissingNormalD(d, missR, n)
		mvnst = MvNormal(zeros(d), 1.0 *I(d))
		L = LowerTriangular(rand(d,d))
		# L = LowerTriangular(Matrix([0.507 0.612; 0.612 0.766]))
		trueμ = rand(d) *3
		Dfull = trueμ .+ L * rand(mvnst, n)
		trueΣ = L * L'
		Data = Matrix{Union{Float64, Missing}}(undef, d, n)
		obsIdx = rand(Bernoulli(missR), d, n) .== 0
		Data[obsIdx] = Dfull[obsIdx]
		return trueμ, trueΣ, Data, Dfull
	end
	d = 6
	missingRate = 0.5
	n_ = 200
	trueμ, trueΣ, D2MVN, D2_=simulateMissingNormalD(d, missingRate, n_)
end

# ╔═╡ ca0b26f1-09a6-4c78-87d4-73bcb2e1e765
begin
	# scatter(D2_[1,:], D2_[2,:])
	tmp=ismissing.(D2MVN) 
	idx_ = (.!any(tmp, dims=1))[:]
	# D2MVN[:, idx_ .==1]
	D2MVNFl =Matrix{Float64}(D2MVN[:, idx_])
	scatter(D2_[1,:], D2_[2,:])
	scatter!(D2MVNFl[1,:], D2MVNFl[2,:])
end

# ╔═╡ 512207ed-67db-4df6-a19e-277bdfd1ecb5
D2MVNFl

# ╔═╡ c7a51d8b-c958-4f4e-aaca-e4520bda7a20
begin 
	function sampleμ(m0, V0, Σ, D; Σinv = missing)
		N = size(D)[2]
		V0inv = inv(V0)
		if ismissing(Σinv)
			Σinv = inv(Σ)
		end
		VnInv = V0inv + N* Σinv
		Vn = inv(VnInv)
		mn = Vn * (V0inv*m0 + Σinv * sum(D, dims=2))
		rand(MvNormal(mn[:], Matrix(Symmetric(Vn))))
	end
	
	
	function sampleΣ(ν0, S0, μ, D)
		N = size(D)[2]
		νn = ν0 + N
		Sμ = (D .- μ) * (D .- μ)'
		Sn = S0 + Sμ
		try 
			rand(InverseWishart(νn, Matrix(Symmetric(Sn))))
		catch e
			display(S0)
			println()
			display(μ)
		end
		rand(InverseWishart(νn, Matrix(Symmetric(Sn))))
	end
	
	function gibbsMVN(D, m0, V0, ν0, S0, mc; μ0 = m0, burnin=0)
		d = size(D)[1]
		μs = zeros(d, mc)
		Σs = zeros(d,d, mc)
		μt = μ0
		for i in 1:(mc+burnin)
			Σt = sampleΣ(ν0, S0, μt, D)
			μt = sampleμ(m0, V0, Σt, D)
			if i > burnin
				Σs[:,:,i-burnin]= Σt
				μs[:,i-burnin] = μt 
			end
		end
		return μs, Σs
	end
end

# ╔═╡ fa5a8e21-d525-4b9f-9fb9-687cda1fbb64
splmvn=gibbsMVN(D2MVNFl, zeros(d), 10. * Matrix(I,d,d), 2+d, Matrix(1.0*I,d,d), 500, μ0 = mean(D2MVNFl, dims=2));

# ╔═╡ 700354d0-ea7c-483c-ade2-a88647764140
(norm(mean(splmvn[1], dims=2) - trueμ), norm(mean(splmvn[2], dims=3) - trueΣ))

# ╔═╡ b5b41d97-ea01-4ac3-9603-ebac8e4567df
begin
	D2MVN_impute = copy(D2MVN)
	for di in 1:d
		D2MVN_impute[di, ismissing.(D2MVN[di,:])] .= mean(skipmissing(D2MVN[di,:]))
	end
	splmvn_impute=gibbsMVN(D2MVN_impute, zeros(d), 10. * Matrix(I,d,d), 2+d, Matrix(1.0*I,d,d), 500, μ0 = mean(D2MVNFl, dims=2));
	(norm(mean(splmvn_impute[1], dims=2) - trueμ), norm(mean(splmvn_impute[2], dims=3) - trueΣ))
end

# ╔═╡ 666b4b80-b430-4703-a4b6-023f3e9c94e7
histogram(sum((trueμ.-splmvn[1]).^2, dims=1)')

# ╔═╡ 8ae90bf3-f194-4b20-9ddb-e3db458c954c
histogram(sum((trueΣ.-splmvn[2]).^2 , dims=(1,2))[:])

# ╔═╡ be727032-5405-4b23-b771-2cac3df0cad3
begin
	function sampleMissingD(d, μt, Λt, missingDim)
		ds = size(Λt)[1]
		mdidx = []
		append!(mdidx, missingDim)
		odidx = setdiff(1:ds, mdidx)
		Λ11 = Λt[mdidx, mdidx]
		Σcon = inv(Λ11)
		μcon = Σcon * (Λ11* μt[mdidx] - Λt[mdidx, odidx] *(d[odidx]- μt[odidx]))
		d[mdidx] = rand(MvNormal(μcon[:], Matrix(Symmetric(Σcon)))) 
		return d
	end
	
	
	function gibbsMissingMVN(D, m0, V0, ν0, S0; μ0 = m0, mc=1000, burnin =1000)
		missingIdx = ismissing.(D) 
		bothMissingObsIdx = all(missingIdx, dims=1)[:]
		Dworking = D[:, (.!bothMissingObsIdx)]
		mIdxDw = ismissing.(Dworking)
		obvIdxDw = (.!any(mIdxDw, dims=1))[:]
		mIdx = any(mIdxDw, dims=1)
		# μt = Matrix{Float64}(mean(Dworking[:, obvIdxDw], dims=2))
		Σt = S0 
		μt = m0
		d = size(D)[1]
		μs = zeros(d, mc)
		Σs = zeros(d, d, mc)
		for i in 1:(mc+burnin)
			Λt = inv(Σt)
	# 		sample missing data
			for j in findall(any(mIdx, dims=1)[:])
				missingDim = findall(mIdxDw[:,j])
				Dworking[:,j] = sampleMissingD(Dworking[:,j], μt, Λt, missingDim)
			end
	# 		sample μ
			μt = sampleμ(m0, V0, Σt, Dworking; Σinv = Λt)
	# 		sample Σ	
			Σt = sampleΣ(ν0, S0, μt, Dworking)
			# μ_, Σ_=gibbsMVN(Dworking, m0, V0, ν0, S0, 1; μ0 = μt);
			
			if i >burnin
				Σs[:,:,i-burnin]= Σt 
				μs[:,i-burnin] = μt
			end
		end
		return μs, Σs
	end
end

# ╔═╡ 0f454f4b-8199-404a-b453-dffbc6456f7b
mm, ss=gibbsMissingMVN(D2MVN, 1.0*zeros(d), 10. *Matrix(I,d,d), 2.0+d, 5.0*Matrix(I,d,d); burnin = 50, mc= 500);

# ╔═╡ 511b1593-0789-4bc1-9c6d-a3abd5a5d134
mean(mm, dims=2)

# ╔═╡ 6815f5f2-fb6d-4751-acda-5e7cc9359bf2
trueμ

# ╔═╡ d7b87606-f560-4ee2-b51f-9559a990efaa
(norm(mean(splmvn_impute[1], dims=2) - trueμ), norm(mean(splmvn_impute[2], dims=3) - trueΣ))

# ╔═╡ 1d83cbf7-04ed-4831-b4d8-1440c9bdaeda
(norm(mean(splmvn[1], dims=2) - trueμ), norm(mean(splmvn[2], dims=3) - trueΣ))

# ╔═╡ 374c15a9-e045-4089-9db5-32b8784992d2
(norm(trueμ - mean(mm, dims=2)), norm(trueΣ - mean(ss, dims=3)))

# ╔═╡ 9791bf1a-eb77-4304-824d-a6b9c09a1139
mean(ss, dims=3)

# ╔═╡ 9e4c23de-2735-4362-a835-9b4de5357e0a
trueΣ

# ╔═╡ 956d26fd-e680-4740-8001-6f5ca74266e7
begin
	histogram(sum((trueμ.-splmvn[1]).^2, dims=1)', label ="Partial Obv", xlabel = "error on μ")
		
	# histogram!(sum((trueμ.-splmvn_impute[1]).^2, dims=1)', label ="Simple Impute")
	histogram!(sum((trueμ.-mm).^2, dims=1)', label ="Full Gibbs")
end

# ╔═╡ 757ccd48-cfda-4812-ba44-4e67e24b453c
begin
	histogram(sum((trueΣ.-splmvn[2]).^2 , dims=(1,2))[:], label="Paritial Obv", xlabel = "error on Σ")
	histogram!(sum((trueΣ.-splmvn_impute[2]).^2 , dims=(1,2))[:], label="Simple Impute")

	
	histogram!(sum((trueΣ.-ss).^2 , dims=(1,2))[:], label ="Full Gibbs")
end

# ╔═╡ 032f7c6c-085c-42b7-a57d-7894514a7207
begin
	function simpleImpute(D, m0, V0, ν0, S0; mc=500, burnin=10)
		D_impute = copy(D)
		d = size(D)[1]
		for di in 1:d
			D_impute[di, ismissing.(D[di,:])] .= mean(skipmissing(D[di,:]))
		end
		spl=gibbsMVN(D_impute, zeros(d), 10. * Matrix(I,d,d), 2+d, Matrix(1.0*I,d,d), mc; μ0 = zeros(d), burnin= burnin);
		return mean(spl[1], dims=2), mean(spl[2], dims=3)
	end
	
	
	function repeatExperiment(times, d, n, missingRate)
		μerrors = zeros(2, times)
		Σerrors = zeros(2, times)
		burnin = 10
		mc = 100
		m0 = 1.0 * zeros(d)
		V0 = 100. *Matrix(I,d,d)
		ν0 = 2 + d
		S0 = 10. * Matrix(I,d,d)
		for t in 1:times
			trueμ, trueΣ, Data, D_=simulateMissingNormalD(d, missingRate, n);
			μimpute, Σimpute = simpleImpute(Data, m0, V0, ν0, S0; burnin = burnin, mc= mc);
			ms, Ss = gibbsMissingMVN(Data, m0, V0, ν0, S0; burnin = burnin, mc= mc);
			μGibbs = mean(ms, dims=2)
			ΣGibbs = mean(Ss, dims=(3))[:,:]
			
			μerrors[1, t] = norm(μimpute - trueμ)
			μerrors[2, t] = norm(μGibbs - trueμ)
			Σerrors[1, t] = norm(Σimpute - trueΣ)
			Σerrors[2, t] = norm(ΣGibbs - trueΣ)
		end
		return μerrors, Σerrors
	end
	
	
	function runExp(dd, nn, times)
		rates = collect(0.05:0.1:0.9)
		ues = zeros(times, 2, length(rates))
		ses = zeros(times, 2, length(rates))
		# dd = 2
		# nn = 50
		ri = 1
		for r in rates
			print(r)
			println()
			ue, se=repeatExperiment(times, dd, nn, r)
			ues[:,:,ri] = ue'
			ses[:,:,ri] = se'
			ri += 1
		end	
		return ues, ses
	end
	ues, ses = runExp(4, 400, 20)
end

# ╔═╡ 3a76b3f9-f4cf-4b5b-93d1-f65b8d4f750f
begin
	rates = collect(0.05:0.1:0.9)
	plot(rates, mean(ses[:,1,:], dims=1)', yerr = std(ses[:,1,:], dims=1)./sqrt(size(ses)[1]), label="Simple Impute", xlabel="Missing Rate", ylabel="L2 error", legend=:bottomright)
	plot!(rates, mean(ses[:,2,:], dims=1)', yerr = std(ses[:,2,:], dims=1)./sqrt(size(ses)[1]), label="Gibbs")
end

# ╔═╡ eae3283e-4c43-4453-9088-b46555fd08e2
begin
	plot(rates, mean(ues[:,1,:], dims=1)', yerr = std(ues[:,1,:], dims=1)./sqrt(size(ues)[1]), label="Simple Impute", xlabel="Missing Rate", ylabel="L2 error", legend=:bottomright)
	plot!(rates, mean(ues[:,2,:], dims=1)', yerr = std(ues[:,2,:], dims=1)./sqrt(size(ues)[1]), label="Gibbs")
end

# ╔═╡ 04ac382d-004e-49ec-b39f-a393be513fee
std(ses[:,1,:], dims=1)

# ╔═╡ 627caeb4-49d5-4a86-8aa6-cd2d44059650
std(ses[:,2,:], dims=1)

# ╔═╡ 91c3b329-68b8-411c-bfcf-3bd7715050fd
md"""
## Question 3
You are given data set $D_3=[d_1, d_2, \ldots, d_T]$, a time series of T blood count observations. Let's assume the data is blood count measurements of some patient over time. The patient has taken some treatment at some unknown point $t_0 \in [1, T)$. Assume his/her blood count changes significantly before and after the treament, which implies you should model the two period's blood counts $D_{30}=\{d_1, \ldots, d_{t_0 -1}\}$ and $D_{31}=\{d_{t_0}, \ldots, d_T\}$ as two Poissons, $\lambda_0$, $\lambda_1$. When did he take the treament, and what is the change of the blood count? 
"""

# ╔═╡ bf3d0a52-14ca-40c3-bc36-f8df7b7b9c19
begin
	T = 200
	thred = 20
	truet0 = rand(thred:(T-thred))
	λ30 = 100
	λ31 = 115
	D3 = Vector{Int}(undef, T)
	D3[1:(truet0-1)] = rand(Poisson(λ30), truet0-1)
	D3[truet0:end] = rand(Poisson(λ31), T-truet0+1)
	plot(D3)
end

# ╔═╡ 47dfd659-7bfc-4de3-a4b9-3ae7219b43ee
begin
	
	function samplet0(D, λ0, λ1)
		T = length(D)
		logpt = zeros(T+1)
		for t in 1:(T+1)
			logpt[t] = sum(logpdf(Poisson(λ0), D3[1:(t-1)])) + sum(logpdf(Poisson(λ1), D[t:end]))
		end
		logsum = logsumexp(logpt)
		pt0 = exp.(logpt .- logsum)
		return rand(Categorical(pt0))
	end
	
	function gibbsChangepoint(D, a0, b0, mc, burnin)
		λts3 = zeros(mc, 3)
		t0 = Int(floor(length(D)/2))
		for i in 1:(mc+burnin)
			# sample λ0
			λ0 = sampleλ(D[1:(t0-1)], a0, b0, 1)[1]
			# sample λ1
			λ1 = sampleλ(D[t0:end], a0, b0, 1)[1]
			# sample t0
			t0 = samplet0(D, λ0, λ1)
			if i > burnin
				λts3[i-burnin,2] = λ0
				λts3[i-burnin,3] = λ1
				λts3[i-burnin,1] = t0
			end
		end
		return λts3
	end
end

# ╔═╡ 11f23275-c6ce-44ae-b7ea-9ba08f6576ea
begin
	λts3 = gibbsChangepoint(D3, 1, 1, 1000, 1000)
	truet0
	plot(Chains(λts3, [:t_0, :λ0, :λ1]))
end

# ╔═╡ 87817df9-9bdf-4ba7-8178-5ad352068932
truet0

# ╔═╡ 68efc1d1-0c64-48e0-b8ac-753027a80ce3
md"""
## Question 4 (Truncated observation)
You are given data set $D_4=[d_1, d_2, \ldots, d_N]$, a sample of N blood count observations. However, the sensor that made the observation is not sensitive when the real count is small. It means there is unknown threshold $h > 0$ such that all observations $d < h$ is not reported. What is this threshold $h$ and what is the true blood count?
"""

# ╔═╡ 9cd743a8-54d0-43b1-982b-f1f83aa1682a
begin
	h = 40
	λ4 = 40
	D4_ = rand(Poisson(λ4), 100)
	D4 = D4_[D4_ .>= h]
	N4 = length(D4)
	histogram(D4, normed=true, label="truncated")
	histogram!(D4_, normed= true, label ="real")

end

# ╔═╡ 39342aa2-588b-4968-8e5d-4b6493257134
begin
# Ported from Radford Neal's R code, with a few thinggs missing

# Arguments:
#
#   x0    Initial point
#   g     Function returning the log of the probability density (plus constant)
#   w     Size of the steps for creating interval (default 1)
#   m     Limit on steps
#   lower Lower bound on support of the distribution (default -Inf)
#   upper Upper bound on support of the distribution (default +Inf)
#   gx0   g(x0)
#
function slice_sampler(x0::Float64, g::Function, w::Float64, m::Int64, lower::Float64, upper::Float64, gx0::Float64)

  if w <= 0
    error("Negative w not allowed")
  end
  if m <= 0
    error("Limit on steps must be positive")
  end
  if upper < lower
    error("Upper limit must be above lower limit")
  end
  
  # Determine the slice level, in log terms.
  
  logy::Float64 = gx0 - rand(Exponential(1.0))

  # Find the initial interval to sample from.

  u::Float64 = rand() * w
  L::Float64 = x0 - u
  R::Float64 = x0 + (w-u)

  # Expand the interval until its ends are outside the slice, or until
  # the limit on steps is reached.  

  J::Int64 = floor(rand() * m)
  K::Int64 = (m-1) - J

  while J > 0
    if L <= lower || g(L)::Float64 <= logy
      break
    end
    L -= w
    J -= 1
  end

  while K > 0
    if R >= upper || g(R)::Float64 <= logy
      break
    end
    R += w
    K -= 1
  end

  # Shrink interval to lower and upper bounds.

  L = L < lower ? lower : L
  R = R > upper ? upper : R
  
  # Sample from the interval, shrinking it on each rejection.

  x1::Float64 = 0.0  # need to initialize it in this scope first
  gx1::Float64 = 0.0
  while true 
    x1 = rand() * (R-L) + L
    gx1 = g(x1)::Float64
    if gx1 >= logy
      break
    end
    if x1 > x0
      R = x1
    else
      L = x1
    end
  end

  return x1,gx1
end
end

# ╔═╡ 3faef526-17a8-4eec-93bc-3d0d170495f6
begin
	function logLikTruncatedPoisson(λ, h, D)
		ch = sum(pdf(Poisson(λ), 0:(h-1)))
		return log(ch) .+ (log.(D .>= h) +logpdf(Poisson(λ), D))
	end
	
	function MHTruncatedPoisson(x0, mc, logp, q)
		spls = zeros(mc)
		xi = x0
		for i in 1:mc
			xstar = rand(q)
			Acp = logp(xstar) + logpdf(q, xi) - logp(xi) - logpdf(q, xstar)
			u = rand()			
			if u < exp(Acp)
				xi = xstar
			end
			spls[i] = xi
		end
		return spls
	end
	
	
	function samplePosth(λ, D)
		hmax = minimum(D)
		ps = zeros(1:hmax)
		for h = 1:length(ps)
			ps[h] = sum(logLikTruncatedPoisson(λ, h, D))
		end
		logsum = logsumexp(ps)
		return rand(Categorical(exp.(ps .- logsum)))
	end
	
	@enum λmethod MHGamma Slice 
	
	function gibbsQ4(D, a0, b0, mc, burnin; method::λmethod = MHGamma, localMC= 20)
		λhs = zeros(mc, 2)
		an = 0.1 * sum(D) + 1.
		bn =  0.1 * length(D) + 1.
		q = Gamma(an, 1/bn)
		h0 = 10
		λ0 = mean(D)
		for i in 1:(mc+burnin)
			# sample λ
			logPost(λ) = logpdf(Gamma(a0,b0), λ) + sum(logLikTruncatedPoisson(λ, h0, D))
			if method == Slice
				for li = 1:localMC
					λ0, _ = slice_sampler(Float64(λ0), logPost, 1., 10000, 0., 500., logPost(λ0))
				end
			elseif method == MHGamma
				λ0 = MHTruncatedPoisson(λ0, 20, logPost, q)[end]
			end
			# sample h
			h0 = samplePosth(λ0, D)
			
			if i > burnin
				λhs[i-burnin, 1] = λ0
				λhs[i-burnin, 2] = h0
			end
		end
		return λhs
	end
end

# ╔═╡ 8e283e2b-b6fb-4cde-8b16-2d20046b22a8
begin
	begin
		mc4 = 1000
		spl4 = gibbsQ4(D4, 1,1, mc4,10; method=Slice, localMC= 10);
		spl4Wrong= sampleλ(D4, 1, 1, mc4)
		plot(Chains(hcat([spl4, spl4Wrong]...), [:λ,:h,:wrongλ]))
	end
end

# ╔═╡ 10ca877d-a976-42c7-9f90-4a343b09ff80
begin
	insupport(λ) = λ > 0
	density(λ) = insupport(λ) ? logPost1(λ) : -Inf
	
	model = DensityModel(density)
	# p1 = StaticProposal(Gamma(50,1))
	# c1 = sample(model, MetropolisHastings(p1), 1000, chain_type= StructArray, param_names=["λ"])
	# plot(c1.λ)
end

# ╔═╡ df7a624a-1143-425d-a204-321300fa53c9
begin
	ts = rand(100)*100
	t0 = 50
	

	# β1 = -1.5
	β0 = 10
	β1 = - β0 /t0
	yt = zeros(length(ts))
	yt[ts .<= t0] = β0 .+ β1 * ts[ts .<= t0] + rand(Normal(0, 1), sum(ts .<=t0))
	
	c= 10
	
	r0 = -log(c/0.1 -1)
	r1 = 1.
	
	ts_ = ts[ts .> t0]
	yt[ts .> t0] = 10. * logistic.(r0 .+ r1 * (ts_ .- t0)) + rand(Normal(0,2), sum(ts .>t0))
	scatter(ts, yt)
end

# ╔═╡ 6d07b66b-b12a-4bad-aa52-e6ba3de52457
begin
	xs = rand(100) 
	xs = hcat([ones(length(xs)), xs]...)
	β = rand(Normal(), 2)
	ys = logistic.(xs * β) + rand(Normal(0, 10), size(xs)[1])
	scatter(xs[:,2], ys)
end

# ╔═╡ 47fbd5c9-fa0a-413a-9ed4-cf379a395451
β

# ╔═╡ Cell order:
# ╠═71d93c0d-0e4a-4684-8a54-6e25c2606a2b
# ╟─797629b6-0669-11ec-3778-d1f20c3d6a4c
# ╟─4570ae10-8419-4af4-aea6-c5fc05756617
# ╠═13cb28fb-6b72-4ccf-9a2e-aec107ccd04b
# ╟─92d4bbd4-a7c1-426a-8c26-a40a4cb37cff
# ╠═d6a367e4-3a76-4a21-85ee-36cafc2a4749
# ╟─493d8a65-744d-4ea2-80d1-01e429643916
# ╠═2aabe7b7-6381-4e78-b0d2-a25aa7d55d68
# ╟─c7d2d072-1551-4901-8cb6-1b222d666b93
# ╠═71a5896b-2e3b-4ba4-8ded-9b46e5ddcbd6
# ╠═369df433-38b1-4dbf-982c-2799302b7d8f
# ╠═ca0b26f1-09a6-4c78-87d4-73bcb2e1e765
# ╠═512207ed-67db-4df6-a19e-277bdfd1ecb5
# ╠═c7a51d8b-c958-4f4e-aaca-e4520bda7a20
# ╠═fa5a8e21-d525-4b9f-9fb9-687cda1fbb64
# ╠═700354d0-ea7c-483c-ade2-a88647764140
# ╠═b5b41d97-ea01-4ac3-9603-ebac8e4567df
# ╠═666b4b80-b430-4703-a4b6-023f3e9c94e7
# ╠═8ae90bf3-f194-4b20-9ddb-e3db458c954c
# ╠═be727032-5405-4b23-b771-2cac3df0cad3
# ╠═0f454f4b-8199-404a-b453-dffbc6456f7b
# ╠═511b1593-0789-4bc1-9c6d-a3abd5a5d134
# ╠═6815f5f2-fb6d-4751-acda-5e7cc9359bf2
# ╠═d7b87606-f560-4ee2-b51f-9559a990efaa
# ╠═1d83cbf7-04ed-4831-b4d8-1440c9bdaeda
# ╠═374c15a9-e045-4089-9db5-32b8784992d2
# ╠═9791bf1a-eb77-4304-824d-a6b9c09a1139
# ╠═9e4c23de-2735-4362-a835-9b4de5357e0a
# ╠═956d26fd-e680-4740-8001-6f5ca74266e7
# ╠═757ccd48-cfda-4812-ba44-4e67e24b453c
# ╠═032f7c6c-085c-42b7-a57d-7894514a7207
# ╠═3a76b3f9-f4cf-4b5b-93d1-f65b8d4f750f
# ╠═eae3283e-4c43-4453-9088-b46555fd08e2
# ╠═04ac382d-004e-49ec-b39f-a393be513fee
# ╠═627caeb4-49d5-4a86-8aa6-cd2d44059650
# ╟─91c3b329-68b8-411c-bfcf-3bd7715050fd
# ╠═bf3d0a52-14ca-40c3-bc36-f8df7b7b9c19
# ╠═47dfd659-7bfc-4de3-a4b9-3ae7219b43ee
# ╠═11f23275-c6ce-44ae-b7ea-9ba08f6576ea
# ╠═87817df9-9bdf-4ba7-8178-5ad352068932
# ╟─68efc1d1-0c64-48e0-b8ac-753027a80ce3
# ╠═9cd743a8-54d0-43b1-982b-f1f83aa1682a
# ╠═3faef526-17a8-4eec-93bc-3d0d170495f6
# ╟─39342aa2-588b-4968-8e5d-4b6493257134
# ╠═8e283e2b-b6fb-4cde-8b16-2d20046b22a8
# ╠═10ca877d-a976-42c7-9f90-4a343b09ff80
# ╠═df7a624a-1143-425d-a204-321300fa53c9
# ╠═6d07b66b-b12a-4bad-aa52-e6ba3de52457
# ╠═47fbd5c9-fa0a-413a-9ed4-cf379a395451
