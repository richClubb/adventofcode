open System.IO

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
    //let path = "../../part_a_sample.txt"
    let path = "../../input.txt"
    //let! lines = File.ReadAllLinesAsync path
    let! content = File.ReadAllTextAsync path

    let blocks = content.Split("\n\n") |> Array.toList
    
    let seed_line = List.filter (fun (block :string) -> block.Contains("seeds: ")) blocks |> List.head
    let seeds = seed_line.Replace("seeds: ", "").Split(" ") |> Array.toList |> List.map (fun seed_string -> int64(seed_string))
    //List.iter (fun (x:Seed) -> printfn $"{x.Value}") seeds

    let map_blocks = List.filter (fun (block :string) -> block.Contains("seeds: ") = false) blocks
    let map_string = List.map (fun (block: string) -> block.Split("\n") |> Array.toList |> List.tail ) map_blocks
    //printfn($"{map_string}")

    let result = List.map (fun x -> convert x) map_string
    //List.iter (fun (x:MappingLayer )-> printfn($"Mapping Layer: "); List.iter (fun (y:Mapping) -> printfn $"{y.src_start}, {y.src_end}, {y.dest_start}, {y.dest_end}") x.Mappings) result

    let fnChain = result[0].TranslateSeedForward >> result[1].TranslateSeedForward >> result[2].TranslateSeedForward >> result[3].TranslateSeedForward >> result[4].TranslateSeedForward >> result[5].TranslateSeedForward >> result[6].TranslateSeedForward    

    let result2 = List.map (fun x -> fnChain x) seeds |> List.min

    printfn($"Part A result: {result2}")

    let seed_ranges_values = Seq.chunkBySize 2 seeds |> Seq.collect Array.pairwise |> Seq.toList

    let seed_ranges = List.map (fun (x, y) -> SeedRange(x, y)) seed_ranges_values

    let result4 = List.map (fun (x: SeedRange) -> x.FindMinSeedInRange result) seed_ranges |> List.min

    printfn($"Part B result {result4}")

}
|> Async.AwaitTask
// we need to run this synchronously
// so the fsi can finish executing the tasks
|> Async.RunSynchronously