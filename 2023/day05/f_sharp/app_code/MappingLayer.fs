namespace app_code

exception TooManyMappings of string

type MappingLayer(mappings: List<Mapping>) = 
    member this.Mappings = mappings

    member this.TranslateSeedForward(seed : int64) = 
        //printfn $"{seed}"
        let mappings = List.filter (fun (mapping: Mapping)-> mapping.CheckSeedInRangeForward seed) this.Mappings
        match mappings.Length with
        | 0 -> seed
        | 1 -> mappings.Head.TranslateSeedForward(seed)
        | _ -> raise (TooManyMappings("Too many mappings"))

    member this.TranslateSeedReverse(seed : int64) = 
        let mappings = List.filter (fun (mapping: Mapping)-> mapping.CheckSeedInRangeReverse seed) this.Mappings
        match mappings.Length with
        | 0 -> seed
        | 1 -> mappings.Head.TranslateSeedReverse(seed)
        | _ -> raise (TooManyMappings("Too many mappings"))

    member this.TranslateSeedsForward(seeds: List<int64>) = 
        List.map (fun seed -> this.TranslateSeedForward(seed)) seeds

    member this.TranslateSeedsReverse(seeds: List<int64>) = 
        List.map (fun seed -> this.TranslateSeedReverse(seed)) seeds