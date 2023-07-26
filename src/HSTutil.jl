#!/usr/bin/env julia
using FITSIO, Plots, Printf, NumericalIntegration, StatsBase, DelimitedFiles, DSP, Interpolations, CubicSplines
#include("pyimports.jl")
using .pyimports: numpy
#sorting all visit files
#files = readdir("/home/kirk/Documents/research/Dexter/STORM/HST/models/")

function getSortedFiles(spectra_dir=pwd())
    files = readdir(spectra_dir)
    function file_sort_key(file_name)
        sub_string = split(file_name, "_")[5]
        file_num = split(sub_string, "-")[end] #visit "number"
        if isnumeric(file_num[3])
            # If the file name ends with a number, sort numerically
            return parse(Int, file_num[2:end])
        else
            # If the file name ends with a letter, add letter position in alphabet to get numerical position (numbers stop at 99)
            return (parse(Int, file_num[2])) + 100 + 26 * (parse(Int, file_num[2])) + Int(file_num[end]) - Int('A') + 1
        end
    end
    sorted_files = sort(files, by=file_sort_key)
    return sorted_files
end

function getFillShape(x, y)
    shapex = vcat(x[1], x[1], x[end], x[end])
    shapey = vcat(y[1], y[end], y[end], y[1])
    return shapex, shapey
end

function getSpectrumTimeInfo(file)
    f = FITS(file)
    header_string = read_header(f[1],String)
    exp = split(split(header_string,"EXPTIME =")[2])[1] #s
    start = split(split(header_string,"HJDSTART=")[2])[1] #heliocentric julian date
    return parse(Float64,exp), parse(Float64,start)
end

struct spectrum
    filename::String
    λ::Array{Float64,1}
    flux::Array{Float64,1}
    error::Array{Float64,1}
    cont_flux::Array{Float64,1}
    model_flux::Array{Float64,1}
    t::Float64
    exp::Float64
end

struct lcDatum #one piece of light curve
    t::Float64 #heliocentric julian Date
    F1158::Float64 #flux at 1158 A
    e_F1158::Float64 #1 sigma error in flux at 1158 A
    F1367::Float64 #flux at 1367 A
    e_F1367::Float64 #1 sigma error in flux at 1367 A
    F1469::Float64 #flux at 1469 A
    e_F1469::Float64 #1 sigma error in flux at 1469 A
    F1745::Float64 #flux at 1745 A
    e_F1745::Float64 #1 sigma error in flux at 1745 A
    F_Lya::Float64 #flux of broad Ly alpha emission
    e_F_Lya::Float64 #1 sigma error in Ly alpha emission
    F_NV::Float64 #flux of broad NV emission
    e_F_NV::Float64 #1 sigma error in NV emission
    F_SiIV::Float64 #flux of broad SiIV emission
    e_F_SiIV::Float64 #1 sigma error in SiIV emission
    F_CIV::Float64 #flux of broad CIV emission
    e_F_CIV::Float64 #1 sigma error in CIV emission
    F_HeII::Float64 #flux of broad HeII emission
    e_F_HeII::Float64 #1 sigma error in HeII emission
end

struct LC
    t::Array{Float64,1} #days
    flux::Array{Float64,1} #flux
    error::Array{Float64,1} #1 sigma error in flux
end

function getLCData(file)
    fileData = readdlm(file,skipstart=30) #first 30 lines are header info pasted above
    data = Array{lcDatum,1}(undef, size(fileData,1)) #initialize data Array
    for i = 1:size(fileData,1)
        data[i] = lcDatum(fileData[i,1], fileData[i,2], fileData[i,3], fileData[i,4], fileData[i,5], fileData[i,6], fileData[i,7], fileData[i,8], fileData[i,9], fileData[i,10], fileData[i,11], fileData[i,12], fileData[i,13], fileData[i,14], fileData[i,15], fileData[i,16], fileData[i,17], fileData[i,18], fileData[i,19])
    end
    return data
end

function getLC(LCData,name,tStop=75.)
    t = [LCData[i].t for i in 1:length(LCData)]
    t = t.-t[1] #normalize to first visit
    if tryparse(Int,name) == nothing
        name = "F_"*name
    else
        name = "F"*name
    end
    f = [getproperty(LCData[i],Symbol(name)) for i in 1:length(LCData)]
    e = [getproperty(LCData[i],Symbol("e_"*name)) for i in 1:length(LCData)]
    mask = t.<tStop
    return LC(t[mask],f[mask],e[mask])
end

#do polynomial interpolation of LC data to fill in between timestamps, specify N points between each tBinned
function getSpline(x::Array{Float64,1},y::Array{Float64,1})
    if length(x) != length(y)
        error("x and y must be same length")
        return nothing
    end
    spline = CubicSpline(x,y)
    f(x) = spline[x]
    return f
end

function getHSTData(LCfile,spectra_dir,line="CIV",cont="1367",tStop=75.,wrap=true)
    LCData = getLCData(LCfile)
    lineLC = getLC(LCData,line,tStop)
    CLC = getLC(LCData,cont,tStop)
    sortedfiles = getSortedFiles(spectra_dir)
    models = getLineModelStrings(line)
    lineData = getSpectra(sortedfiles,src_dir=spectra_dir,model=models)
    avgSpectrum = getAvgSpectrum(lineData,λRange=[1450,1725])
    if wrap
        return lineLC.t,lineLC.flux,lineLC.error,CLC.t,CLC.flux,avgSpectrum.λ,avgSpectrum.model_flux,avgSpectrum.error
    else
        return lineLC,CLC,avgSpectrum
    end
end

function getHSTDataWrap(input)
    LCfile,spectra_dir,line,cont,tStop = input
    if typeof(tStop) != Float64
        tStop = tryparse(Float64,tStop)
    end
    line_t,lineLC,LCerror,cont_t,contLC,λ,model_flux,LPerror = getHSTData(LCfile,spectra_dir,line,cont,tStop)
    return numpy[].array([line_t,lineLC,LCerror,cont_t,contLC,λ,model_flux,LPerror],dtype="object")
end

function plotLC(LCData, lineList; tRange = nothing, spline = false, model = nothing)
    t = [LCData[i].t for i in 1:length(LCData)]
    t = t.-t[1] #normalize to first visit
    mask = [true for i in 1:length(LCData)]
    p = plot(title = "HST STORM Light Curves",xlabel="t [days after first visit]", ylabel="Flux [normalized (1e-15 continuum; 1e-13 line)]", legend=:topleft)
    if tRange != nothing
        mask = (t.>tRange[1]) .& (t.<tRange[2])
        t = t[mask]
    end
    for (i,line) in enumerate(lineList)
        f = [getproperty(LCData[i],Symbol(line)) for i in 1:length(LCData)]
        norm = line == "F1367" ? 1e-15 : 1e-13 #normalize to 1e-15 for continuum, 1e-13 for emission lines
        e = [getproperty(LCData[i],Symbol("e_"*line)) for i in 1:length(LCData)]
        p = plot!(t,f[mask]./norm,ribbon=e[mask]./norm,marker=:circle,markerstrokewidth=0.,label=line,lw = spline ? 0. : 2.,linealpha = spline ? 0. : 1.,c=i)
        if spline
            interp = getSpline(t,f[mask]./norm)
            tInterp = range(t[1],t[end],length=1000)
            p = plot!(tInterp,interp.(tInterp),label="",lw=2.,linealpha=1.,c=i)
        end
        if model != nothing && line != "F1367" #assume model = [model_continuum_LC, model_emission_LC]
            data_span = maximum(f) - minimum(f)
            f_model = (model[2].*data_span .+ f[1])./norm #zero point is first flux value, match span (initial span should be 1)
            if length(t) != length(f_model)
                maxInd = minimum([length(t),length(f_model)])
                t_model = t[1:maxInd]; f_model = f_model[1:maxInd]
            end
            p = plot!(t_model,f_model,marker=:star,markerstrokewidth=0.,label=line*" model",lw = spline ? 0. : 2.,linealpha = spline ? 0. : 1.,c=i,ls=:dash)
            if spline
                interp = getSpline(t_model,f_model)
                tInterp = range(t_model[1],t_model[end],length=1000)
                p = plot!(tInterp,interp.(tInterp),label="",lw=2.,linealpha=1.,c=i,ls=:dash)
            end
        end
    end
    p = plot!(ylims=(0,100),size=(720,720),minorticks=true,widen=false)
    return p
end

function syntheticLC(Ct,CLC,Ψτ;continuum="1367",tStop=75.,spline=false) #assumes Ψτ is already interpolated to continuum timestamps as in functions.jl/getProfiles
    Ψτ = Ψτ./maximum(Ψτ) #normalize
    #LC(t) = integral_0^t Ψτ(t-τ)*C(τ)dτ where C(τ) is the continuum light curve
    #write function that does integral above and use to get synthetic light curve
    #C = getLC(LCData,continuum)
    #C needs to be interpolated to match tBinned -- actually instead let's interpolate Ψτ and tBinned to match continuum t
    #in reality later in modelling we can just sample tBinned at continuum t but for simple exercise do this
    t = Ct
    mask = t.<tStop
    t = t[mask]; C = CLC[mask]; Ψτ = Ψτ[mask] #only go to tStop
    C = C./maximum(C)
    span = maximum(C) - minimum(C) 
    ΔC = C .- C[1]
    ΔC = ΔC ./ span #make span 1, normalize to first point so that + = brighter and - = dimmer
    if spline 
        CSpline = getSpline(t,ΔC)
        ΨτSpline = getSpline(t,Ψτ)
        t = range(t[1],t[end],length=1000)
        ΔC = CSpline.(t); Ψτ = ΨτSpline.(t)
    end
    function LC(t,Ψτ,ΔC)
        N = length(t)
        LC = zeros(N)
        for i=1:N
            ti = t[i]
            integrand = 0
            for τ = 1:i
                dτ = i < N ? t[τ+1]-t[τ] : t[end] - t[end-1]
                integrand += Ψτ[i-τ+1]*ΔC[τ]*dτ
            end
            LC[i] = integrand
        end
        return LC #this is really like ΔLC
    end
    Δlc = LC(t,Ψτ,ΔC) #these are variations, not properly normalized
    spanlc = maximum(Δlc) - minimum(Δlc) #when normalized the extent of variations in continuum should be the same as in emission line
    return Δlc./spanlc,ΔC #checked by doing CCF and this produces same thing as data!
end

#now generate cross correlation function from getSpectra
function getCCF(LCData; continuum="F1367", emission="F_Lya", tRange = nothing, normalize=true, lags = nothing, spline = false, model = nothing)
    #continuum and emission are the names of the fields in the datum struct
    #spectra is an array of datum structs
    #lags can either be an integer array of index shifts to try, a minimum and maximum time shift, or nothing (default)
    #returns an array of cross correlation values
    
    #TO-DO -- add interpolation in case spectra t and continuum t don't match up -- or is this already done? confused by paper vs. txt file
    mask = Bool.(ones(Int,length(LCData)))
    t = [LCData[i].t for i in 1:length(LCData)]
    t = t.-t[1] #normalize to first visit
    if tRange != nothing
        mask = (t.>tRange[1]) .& (t.<tRange[2])
        t = t[mask]
    end
    u = [getproperty(LCData[i],Symbol(continuum)) for i in 1:length(LCData)]
    v = [getproperty(LCData[i],Symbol(emission)) for i in 1:length(LCData)]
    if spline 
        uSpline = getSpline(t,u[mask])
        vSpline = getSpline(t,v[mask])
        tSpline = range(t[1],t[end],length=1000)
        u = uSpline.(tSpline); v = vSpline.(tSpline)
    end
    if normalize
        u = u./maximum(u)
        v = v./maximum(v)
    end
    if lags == nothing || (typeof(lags) != Array{Float64,1} && typeof(lags) != Array{Int64,1}) #default lags from docs
        lags = collect(-minimum(Int,(size(u,1)-1, floor(10*log10(size(u,1))))):minimum(Int,(size(u,1), floor(10*log10(size(u,1))))))
    elseif length(lags) == 2 #lagRange
        minInd = -findfirst(t.>abs(lags[1]))
        maxInd = findfirst(t.>lags[2])
        lags = collect(minInd:maxInd)
    end
    CCF = crosscor(u,v,lags) #cross correlation function
    if model != nothing
        u = model[1]; v = model[2]
        if length(t) != length(u)
            maxInd = minimum([length(t),length(u)])
            u = u[1:maxInd]; v = v[1:maxInd]; t = t[1:maxInd]
        end
        if spline
            uSpline = getSpline(t,u)
            vSpline = getSpline(t,v)
            u = uSpline.(t); v = vSpline.(t)
        end
        model_ccf = crosscor(u,v,lags)
        CCF = [CCF,model_ccf] #also return model CCF, assume model = [model_continuum_LC, model_emission_LC]
    end
    #lags[1] = 0 -> t[1]; lags[end] = max +shift -> t[lags[end]+1]; lags[1] = max -shift -> -t[lags[1]+1]
    return CCF, vcat(-reverse(t[1:abs(lags[1])+1]),t[2:lags[end]+1]) #lags here are index offsets from 0
end #trying to reproduce figure 4 in https://iopscience.iop.org/article/10.1088/0004-637X/806/1/128/pdf , still not working perfectly

function plotCCF(LCData; continuum="F1367",emission="F_Lya",tRange=nothing,normalize=true,lags=nothing,spline=false,model=nothing)
    CCF, t = getCCF(LCData,continuum=continuum,emission=emission,tRange=tRange,normalize=normalize,lags=lags,spline=spline,model=model)
    p = plot()
    if length(CCF) == 1
        p = plot!(t,CCF,legend=false,xlabel="offset τ [days]",ylabel="r",ylims=(-0.1,1),widen=false,size=(720,720),minorticks=true,
            linewidth=2.,c=3,marker=:circle,markerstrokewidth=0.,title="$emission delays relative to $continuum",label="",yticks=[0.2*i for i=0:5])
        τpeak = t[findmax(CCF)[2]]
        p = vline!([τpeak],label="peak delay = $(round(τpeak,digits=2)) days",c=:crimson,lw=2,ls=:dash,legend=:topright)
    else
        labels=["data","model"]
        for (i,ccf) in enumerate(CCF)
            p = plot!(t,ccf,legend=false,xlabel="offset τ [days]",ylabel="r",ylims=(-0.1,1),widen=false,size=(720,720),minorticks=true,
                linewidth=2.,c=i+2,marker=:circle,markerstrokewidth=0.,title="$emission delays relative to $continuum",label=labels[i],yticks=[0.2*i for i=0:5])
            τpeak = t[findmax(ccf)[2]]
            p = vline!([τpeak],label="peak delay = $(round(τpeak,digits=2)) days",c=i+2,lw=2,ls=:dash,legend=:topright)
        end
    end
    if lags != nothing
        p = plot!(xlims=lags)
    end

    return p
end

function LC_CCF_stacked(LCData; continuum="F1367",emission="F_Lya",tRange=nothing,normalize=true,lags=nothing,spline=false,model=nothing)
    p1 = plotLC(LCData, [continuum,emission],tRange=tRange,spline=spline,model=model)
    p2 = plotCCF(LCData,continuum=continuum,emission=emission,tRange=tRange,normalize=normalize,lags=lags,spline=spline,model=model)
    p = plot(p1,p2,layout=@layout([a;b]),size=(720,1440),left_margin=10*Plots.Measures.mm)
    return p
end

function getLineModelStrings(line::String; sample_fits::String="/home/kirk/Documents/research/Dexter/STORM/HST/models/hlsp_storm_hst_cos_ngc-5548-go13330-v0a_g130m-g160m_v1_model.fits") #don't think this is quite right at end product, do further testing tmoorrow
    f = FITS(sample_fits)
    header_string = read_header(f[1],String)
    split_strip_header = strip.(split(header_string,"HISTORY"))
    GaussianModelStrings = split_strip_header[startswith.(split_strip_header,"Gaussian")]
    lineMask = occursin.(line,GaussianModelStrings)
    lineStrings = GaussianModelStrings[lineMask]
    BLRMask = occursin.("Broad",lineStrings)
    BLAMask = occursin.("Broad Absorption",lineStrings)
    BLREmissionMask = (BLRMask) .& (.!BLAMask)
    bumpMask = occursin.("Bump",lineStrings)
    finalMask = (BLREmissionMask) .|| (bumpMask)
    lineStrings = lineStrings[finalMask]
    components = [strip(split(lineStrings[i]," ")[3],':') for i in eachindex(lineStrings)]
    return tryparse.(Int,components)
end
    #remove "broad absorption" lines from getLineModelStrings

    
function getSpectra(sortedFiles; src_dir=pwd(),model="all")
    data = Array{spectrum,1}(undef, length(sortedFiles))
    for (i,file) in enumerate(sortedFiles)
        filename = string(src_dir,file)
        f = FITS(filename)
        λ = read(f[2], "WAVE") #angstroms
        flux = read(f[2], "FLUX") #cW / m^2 / A
        error = read(f[2], "e_FLUX")#"          "
        cont_flux = read(f[2], "FCONT") #"      "
        model_flux = nothing
        if model == "all"
            model_flux = read(f[2], "MODFLUX")
        else
            if typeof(model) == Array{Int64,1}
                model_flux = read(f[2],"GAUSS$(model[1])")
                for j in model[2:end]
                    model_flux .+= read(f[2],"GAUSS$j")
                end
            elseif typeof(model) == Int64 || typeof(model) == Int32
                model_flux = read(f[2],"GAUSS$model")
            else
                error("model must be an integer or array of integers")
            end
        end
        exp,t = getSpectrumTimeInfo(filename)
        data[i] = spectrum(filename, λ, flux, error, cont_flux, model_flux, t, exp)
    end
    return data
end

struct avgSpectrum
    λ::Array{Float64,1}
    flux::Array{Float64,1}
    error::Array{Float64,1}
    model_flux::Array{Float64,1}
end

function getAvgSpectrum(spectra; λRange=nothing)
    avg_λ = sort(unique(vcat([spectra[i].λ for i in 1:length(spectra)]...)))
    if λRange != nothing
        mask = (avg_λ .> λRange[1]) .& (avg_λ .< λRange[2])
        avg_λ = avg_λ[mask]
    end
    avg_flux = zeros(length(avg_λ))
    avg_error = zeros(length(avg_λ))
    avg_model_flux = zeros(length(avg_λ))
    n = zeros(length(avg_λ))
    for (i,λ) in enumerate(avg_λ)
        for spectrum in spectra
            if λ in spectrum.λ
                ind = findfirst(spectrum.λ .== λ)
                avg_flux[i] += spectrum.flux[ind]
                avg_error[i] += spectrum.error[ind]^2
                avg_model_flux[i] += spectrum.model_flux[ind]
                n[i] += 1
            end
        end
    end
    return avgSpectrum(avg_λ,avg_flux./n,sqrt.(avg_error)./n,avg_model_flux./n)
end