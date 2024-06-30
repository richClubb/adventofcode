namespace app_code

open System

type SeedRange(start : int64, size : int64) = 
    member this.Seeds = [ start .. start + size - int64(1) ]

    member this.FindMinSeedInRange( mappingLayers : List<MappingLayer> ) = 

        let mappingFn = mappingLayers[0].TranslateSeedForward 
                        >> mappingLayers[1].TranslateSeedForward 
                        >> mappingLayers[2].TranslateSeedForward 
                        >> mappingLayers[3].TranslateSeedForward 
                        >> mappingLayers[4].TranslateSeedForward 
                        >> mappingLayers[5].TranslateSeedForward 
                        >> mappingLayers[6].TranslateSeedForward

        let results = List.map (fun x -> mappingFn(x)) this.Seeds
        let minSeed = List.min results

        minSeed