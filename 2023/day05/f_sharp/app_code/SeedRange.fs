namespace app_code

open System


type SeedRange(start : int64, size : int64) = 
    member this.SeedStart: int64 = start
    member this.Size: int64 = size
    member this.SeedEnd: int64 = start + size
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

module tools = 
    let split_range (seed_range: SeedRange) (new_max_size: int64) : SeedRange array =

        if seed_range.Size <= new_max_size then 
            let new_seed_ranges: SeedRange array = [|seed_range|]
            new_seed_ranges
        elif (seed_range.Size / new_max_size) = int64(1) then
            let start_1 = seed_range.SeedStart
            let size_1 = seed_range.Size
            let start_2 = seed_range.SeedStart + new_max_size
            let size_2 = seed_range.Size - new_max_size
            let new_seed_ranges: SeedRange array = [|SeedRange(start_1, size_1); SeedRange(start_2, size_2)|]
            new_seed_ranges
        
        let new_seed_ranges : SeedRange array = []
        new_seed_ranges
