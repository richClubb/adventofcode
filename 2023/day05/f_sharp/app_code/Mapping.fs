namespace app_code

type Mapping(dest : int64, src : int64, size : int64) = 
    member this.src_start = src
    member this.src_end = src + size - int64(1)
    member this.dest_start = dest
    member this.dest_end = dest + size - int64(1)

    member this.TranslateSeedForward (seed: int64): int64 = 
        match seed with
        | seed_value when seed_value >= this.src_start && seed_value <= this.src_end -> this.dest_start + seed_value - this.src_start
        | _ -> seed

    member this.TranslateSeedReverse (seed: int64): int64 = 
        match seed with
        | seed_value when seed_value >= this.dest_start && seed_value <= this.dest_end -> this.src_start + seed_value - this.dest_start
        | _ -> seed

    member this.CheckSeedInRangeForward (seed: int64) : bool =
        match seed with
        | seed_value when seed_value >= this.src_start && seed_value <= this.src_end -> true
        | _ -> false

    member this.CheckSeedInRangeReverse (seed: int64) : bool =
        match seed with
        | seed_value when seed_value >= this.dest_start && seed_value <= this.dest_end -> true
        | _ -> false