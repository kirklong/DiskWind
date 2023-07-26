module DiskWind
    function __init__()
        include("pyimports.jl")
        include("functions.jl")
        include("HSTutil.jl")  
    end
    export getProfiles, readPickle, getLCData, getSpectra, getHSTDataWrap
end
