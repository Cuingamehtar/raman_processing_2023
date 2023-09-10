import Pkg; Pkg.activate(mktempdir())
Pkg.add(["DrWatson", "Logging", "DelimitedFiles", "Polynomials", "Interpolations", "Optim", "Statistics", "StatsBase", "Measurements", "Images", "Dates"])
using DrWatson
@quickactivate "Raman"
using Raman

using Logging
using DelimitedFiles, Polynomials, Interpolations
using Optim
using Statistics, StatsBase
using Measurements
using Images

function Measurements.:±(v::AbstractVector{T}) where {T}
    m, s = mean_and_std(v)
    m ± s
end 

kern = let
	k = ImageFiltering.Kernel.gaussian(5)[:,0]
	k ./= sum(k)
	k
end

"""
██      ███████ ███████  █████  
██      ██      ██      ██   ██ 
██      █████   ███████ ███████ 
██      ██           ██ ██   ██ 
███████ ███████ ███████ ██   ██ 
"""

base_x_range = (300, 360)

ppix_fluo = let
    data = readdlm("filtered_ppix_fluo.txt", '\t')
    x = data[:,1]
    y = data[:,2]
    y ./= maximum(y)
    LinearInterpolation(x, y)
end

function get_oxy()
    oxy_file = readdlm("oxygenation.txt", '\t')
    f = x -> convert.(Float64, x)
    data = f.(oxy_file[2:end, :])
    oxy_plus = LinearInterpolation(data[:, 1], data[:, 2])
    oxy_minus = LinearInterpolation(data[:, 1], data[:, 3])
    return oxy_plus, oxy_minus
end
oxy_plus, oxy_minus = get_oxy()

function rescale_axis(x, xp, fp, order = length(xp)-1)
    @assert length(xp) == length(fp) "Scale: Different number of marker points"
    
    pol = Polynomials.fit(xp, fp, order)
    pol.(x)
end

struct LesaData
    x
    white
    abs
    fluo
    fluo_expo
end

function find_center(x,y)
	i_max = argmax(y)
	y_max = y[i_max]

	bound = y_max * 0.2

	idx = falses(size(x))
	idx[i_max] = true

	for i in (i_max + 1):length(y)
		if y[i] > bound
			idx[i] = true
		else
			break
		end
	end

	for i in (i_max - 1):-1:1
		if y[i] > bound
			idx[i] = true
		else
			break
		end
	end

	return mean(x[idx], Weights(y[idx]))
end

"""
┌─┐┬  ┬ ┬┌─┐
├┤ │  │ ││ │
└  ┴─┘└─┘└─┘
"""
function integralintensity(x, y, edges)
    mask = first(edges) .≤ x .≤ last(edges)
	return area(x[mask], y[mask])
end

function fluorescenceindex(x, y)
    laser = integralintensity(x, y, 625:640)

    ppix_fluo_range = ppix_fluo.itp.knots[1][1], ppix_fluo.itp.knots[1][end]
    ppix_mask = ppix_fluo_range[1] .< x .< ppix_fluo_range[2]
    ppix_x = x[ppix_mask]
    ppix_y = ppix_fluo.(ppix_x)
    sub_y = y[ppix_mask]
    
    function lossbelow(a)
         resid = sub_y .- ppix_y*a[1]
         map(r -> r < 0 ? (r*100)^2 : r^2, resid) |> sum
    end

    fit_res = optimize(lossbelow, [0.0], [Inf], [1.0],  Fminbox(NelderMead()))
    fluo = fit_res.minimizer[1]

    # standard implementation
    # fluo = integralintensity(x, y, 690:730)
    return fluo / laser
end

function fluorescenceindex_FAD(x, y)
    laser = integralintensity(x, y, 390:420)
    fluo = integralintensity(x, y, 460:560)
    return fluo / laser
end

function fluorescenceindex_porph(x, y)
	lims = (600, 730)

    mask = 600 .< x .< 730

    function curve(P)
		a, b = P
		res = a .* x[mask] .+ b .- y[mask]
		map(v -> v > 0 ? (50v)^2 : v^2, res) |> sum
	end
    base = optimize(curve, [-0.01, 0.0], NelderMead())
    a = base.minimizer[1]
	b = base.minimizer[2] 

    laser = integralintensity(x, y, 390:420)
    fluo = integralintensity(x[mask], y[mask] .- (a .* x[mask] .+ b), first(lims):last(lims))
    #=
    
	yleft = y[findfirst(i -> x[i] > first(lims), eachindex(x))]
	xleft = x[findfirst(i -> x[i] > first(lims), eachindex(x))]
	yright = y[findlast(i -> x[i] < last(lims), eachindex(x))]
	xright = x[findlast(i -> x[i] < last(lims), eachindex(x))]
	base = LinearInterpolation([xleft, xright], [yleft, yright], extrapolation_bc=Line())
    fluo = integralintensity(x, y .- base.(x), first(lims):last(lims))
    =#
    return fluo / laser
    
end

function backscatter_633(x, y)
    laser = integralintensity(x, y, 625:640)
    return laser
end

function backscatter_405(x, y)
    laser = integralintensity(x, y, 390:420)
    return laser
end

function applylesa(f, lesa::LesaData; normalize=false)
    fi = map(enumerate(lesa.fluo)) do (i,y)
        yn = normalize ? y ./ lesa.fluo_expo[i] : y 
        return f(lesa.x, yn)
    end
    return ±fi
end

function applylesawhitesm(f, lesa::LesaData; normalize=false)
    fi = map(enumerate(lesa.fluo)) do (i,y)
        yn = normalize ? y ./ lesa.fluo_expo[i] : y
        yn = ImageFiltering.imfilter(yn, kern)
        w_itp = LinearInterpolation(x_white_405, white_405; extrapolation_bc=Interpolations.Flat())
        return f(lesa.x, yn ./ w_itp)
    end
    return ±fi
end

"""
┌─┐┌┐ ┌─┐
├─┤├┴┐└─┐
┴ ┴└─┘└─┘
"""
function scatter_hemoglobin(lesa::LesaData)
    if isnothing(lesa.abs)
        return (hemo = missing, scat = missing)
    end
    edges = 500:596
    mask = first(edges) .≤ lesa.x .≤ last(edges)
    fit_results = map(lesa.abs) do y
        y_abs = -log10.(max.(y[mask], 0.0001) ./ lesa.white[mask])

        scatter_hemoglobin(lesa.x[mask], y_abs)
    end

    hemo = map(fr -> fr.hemoglobin, fit_results)
    scat = map(fr -> fr.scatter, fit_results)
    (hemo = ±hemo, scat = ±scat)
end

sigmoid(x) = 1 / (1 + exp(-x))
function mie_scattering(X, P)
    A, f, b = P
    s = sigmoid(f)
    map(X) do x
        return A * (s * x^(-b) + (1 - s) * x^(-4))
    end
end

function diffusescattermodel(X, P)
    A, f, b, c_oxy, c_deoxy, C = P
    mie_scattering.(X, Ref((A,f,b))) .+ c_oxy .* oxy_plus.(X) .+ c_deoxy .* oxy_minus.(X) .+ C
end

function scatter_hemoglobin(x, y)
    opt_f(P) = loss(X -> diffusescattermodel(X, P), x, y)
    fit_res = optimize(opt_f, 
    [0.0, -Inf, 0.1, 0.0, 0.0, -Inf], 
    [1000.0, Inf, 3.0, 10.0, 10.0, 5.0],
    [0.1, 0.0, 1.0, 1.0, 1.0, 0.0],  Fminbox(NelderMead()))
    # println(vcat(fit_res.minimizer[1], sigmoid(fit_res.minimizer[2]), fit_res.minimizer[3:end]))
    (hemoglobin = sum(fit_res.minimizer[4:5]), scatter=fit_res.minimizer[1]) 
end


loss(f, X, Y) = sum(((Y .- f(X)) ./ Y) .^ 2)


"""
██████   █████  ███    ███  █████  ███    ██ 
██   ██ ██   ██ ████  ████ ██   ██ ████   ██ 
██████  ███████ ██ ████ ██ ███████ ██ ██  ██ 
██   ██ ██   ██ ██  ██  ██ ██   ██ ██  ██ ██ 
██   ██ ██   ██ ██      ██ ██   ██ ██   ████ 
"""

"""
┌─┐┌─┐┌─┐┬┌─┌─┐
├─┘├┤ ├─┤├┴┐└─┐
┴  └─┘┴ ┴┴ ┴└─┘
"""

raman_peaks = [
    :cholesterol  => [925, 961, 1060:1066, 1126:1133, 1174, 1228, 1296:1302, 1439:1455, 1669:1674, 1731:1739],
    :phospholipid => [1060:1066, 1080:1090, 1126:1133, 1260, 1296:1302],
    :lipid        => [1088, 1142, 1302:1305, 1439:1455],
    :carothenoid  => [956, 1002:1006, 1157:1159, 1518:1527],
    :heme         => [1002:1006, 1124, 1231:1258, 1346, 1454,1546:1568, 1585],
    :oxy_heme     => [1212, 1603:1605, 1619:1626],
    :protein      => [1042:1049, 1088:1097, 1126:1133, 1174:1175, 1313, 1346, 1397, 1439:1455, 1466, 1575:1588, 1614],
    :phenylalanine=> [1002:1006, 1032, 1206:1208]
]

function area(x, y)
    dx = x[2:end] .- x[1:end-1]
    s = sum((y[1:end-1] .+ y[2:end]) .* dx)/2
end

function chem_index(s, chem)
    x = map(v->Energy(v, Wavelength(785)).value, s.x)
    y = s.y
    s_interp = LinearInterpolation(x, y)
    np = length(chem[2])
    s = 0.0
    for p in chem[2]
        if (first(p) > last(x)) || (last(p) < first(x))
            continue
        end
        if p isa Number
            s += s_interp(p)
        elseif p isa UnitRange
            mask = first(p) .< x .< last(p)
            sub_x = vcat(first(p), x[mask], last(p))
            sub_p1 = s_interp.(sub_x)
            s += area(sub_x, sub_p1) / (last(p) - first(p))

        end
    end
    s / np
end

process_raman_spectrum(::Missing; args...) = missing
function process_raman_spectrum(s::Spectrum, smoother, baseline, spectrum_subsection)
    s = smoother(s)
    s = s - baseline(s)
	x = map(v -> Raman.Energy(v, Wavelength(785.0)).value, s.x)
    mask = spectrum_subsection[1] .< x .< spectrum_subsection[2]
    x = x[mask]
    y = s.y[mask] .- minimum(s.y[mask])
    y ./= area(x, y)
    s = Spectrum(Raman.Energy.(x), y)
	
	map(raman_peaks) do p
		p[1] => chem_index(s, p[2]) * 1000 # scaling coefficient
	end
end

function prepare_sample_raman(sample_directory)
	s = Raman.get_average_raman_spectrum(sample_directory)
	smoother = SavitzkyGolaySmoothing(; width=15, order=3)
	baseline = WaveletBaseline(; λ=1000, scales=1:70)
	spectrum_subsection = 800, 1700
	process_raman_spectrum(s, smoother, baseline, spectrum_subsection)
end