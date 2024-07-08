namespace app_code

open System

type SeedRange(start : int64, size : int64) = 
    member this.Seeds = [ start .. start + size - int64(1) ]

    // This works in a more traditionally functional way, but eats memory.
    member this.FindMinSeedInRange( mappingLayers : List<MappingLayer> ) = 

        let processed_values: int64 list list = List.map (fun seed -> (seed, mappingLayers) ||> List.scan (fun s v -> v.TranslateSeedForward(s))) this.Seeds
        List.map (fun a -> List.last a) processed_values |> List.min

    // This works by only keeping track of the minimum value, rather than trying to store the list of all values
    member this.FindMinSeedInRangeMutable( mappingLayers : List<MappingLayer> ) = 

        let mutable min_value = Int64.MaxValue
        for seed in this.Seeds do
            let seed_val = (seed, mappingLayers) ||> List.scan (fun s v -> v.TranslateSeedForward(s)) |> List.last
            if seed_val < min_value then min_value <- seed_val
        min_value