namespace app_code

open System

type SeedRange(start : int64, size : int64) = 
    member this.SeedStart = start
    member this.SeedEnd = start + size - int64(1)
    // This works in a more traditionally functional way, but eats memory.
    member this.FindMinSeedInRange( mappingLayers : List<MappingLayer> ) = 

        let seeds = [|this.SeedStart .. this.SeedEnd|]
        let processed_values = Array.map (fun seed -> (seed, mappingLayers) ||> List.scan (fun s v -> v.TranslateSeedForward(s))) seeds
        Array.map (fun a -> List.last a) processed_values |> Array.min

    // This works by only keeping track of the minimum value, rather than trying to store the list of all values
    // by using a while loop rather than other iterators it means that it doesnt eat loads of memory
    member this.FindMinSeedInRangeMutable( mappingLayers : List<MappingLayer> ) = 

        let mutable min_value = Int64.MaxValue
        let mutable seed = this.SeedStart
        while seed <= this.SeedEnd do 
            let seed_val = (seed, mappingLayers) ||> List.scan (fun s v -> v.TranslateSeedForward(s)) |> List.last
            if seed_val < min_value then min_value <- seed_val
            seed <- seed + int64(1)
        min_value