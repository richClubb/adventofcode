open System.IO
open FSharp.Collections

open app_code

// For more information see https://aka.ms/fsharp-console-apps
printfn "Advent of Code day 05 in F#"

let extract (athing:string) = 
    let splitString = athing.Split(" ") |> Array.toList
    let ints = List.map (fun (x:string) -> int64(x)) splitString
    Mapping(ints[0], ints[1], ints[2])

let convert (athing: string list) =
    let result = List.map (fun x -> extract x) athing
    MappingLayer(result)

task {
    // Uncomment the line for the file you want. Haven't got command line params working yet.
    let path = "../../input.txt"
    //let path = "../../part_a_sample.txt"

    let! content = File.ReadAllTextAsync path
    let blocks = content.Split("\n\n") |> Array.toList
    
    let seed_line = List.filter (fun (block :string) -> block.Contains("seeds: ")) blocks |> List.head
    let seeds = seed_line.Replace("seeds: ", "").Split(" ") |> Array.toList |> List.map (fun seed_string -> int64(seed_string))

    let map_blocks = List.filter (fun (block :string) -> block.Contains("seeds: ") = false) blocks
    let map_string = List.map (fun (block: string) -> block.Split("\n") |> Array.toList |> List.tail ) map_blocks
    let map_layers = List.map (fun x -> convert x) map_string
    
    // Part A 
    let locations: int64 list list = List.map (fun seed -> (seed, map_layers) ||> List.scan (fun s v -> v.TranslateSeedForward(s))) seeds
    let part_a_min_seed = List.map (fun a -> List.last a) locations |> List.min

    printfn($"Part A result: {part_a_min_seed}")

    //This doesn't compute fast enough for the main input sample
    let seed_ranges_values = Seq.chunkBySize 2 seeds |> Seq.collect Array.pairwise |> Seq.toList
    let seed_ranges = List.map (fun (x, y) -> SeedRange(x, y)) seed_ranges_values
    
    // This doesn't work as it eats such a large amount of memory it just crashes
    // let part_b_min_seed = List.map (fun (x: SeedRange) -> x.FindMinSeedInRangeMutable map_layers) seed_ranges |> List.min

    // This works. Takes about 80 minutes
    let part_b_min_seed = List.map (fun (x: SeedRange) -> x.FindMinSeedInRangeMutable map_layers) seed_ranges |> List.min
    printfn($"Part B result {part_b_min_seed}")

}
|> Async.AwaitTask
// we need to run this synchronously
// so the fsi can finish executing the tasks
|> Async.RunSynchronously