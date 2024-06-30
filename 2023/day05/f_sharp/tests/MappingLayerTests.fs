namespace tests

open System
open Microsoft.VisualStudio.TestTools.UnitTesting
open app_code

[<TestClass>]
type MappingLayerTests () =

    [<TestMethod>]
    member this.TestListFilter () =
        let list = [1; 2; 3]
        let only_2 = List.filter (fun x -> x = 2) list
        Assert.AreEqual(2, only_2.Head)

    [<TestMethod>]
    member this.TestMappingLayerConstruction() =
        let mappingLayer = MappingLayer([Mapping(5, 15, 5); Mapping(50, 60, 3)])
        Assert.AreEqual(2, mappingLayer.Mappings.Length)

    [<TestMethod>]
    member this.TestMappingLayerSeedOutOfRangge() =
        let mappingLayer = MappingLayer([Mapping(5, 15, 5); Mapping(50, 60, 3)])
        let seed = int64(2)
        let result = mappingLayer.TranslateSeedForward(seed)
        Assert.AreEqual(int64(2), result)

    [<TestMethod>]
    member this.TestMappingLayerSeedInRange1() =
        let mappingLayer = MappingLayer([Mapping(5, 15, 5); Mapping(50, 60, 3)])
        let seed = int64(16)
        let result = mappingLayer.TranslateSeedForward(seed)
        Assert.AreEqual(int64(6), result)

    [<TestMethod>]
    member this.TestMappingLayerSeedInRange2() =
        let mappingLayer = MappingLayer([Mapping(5, 15, 5); Mapping(50, 60, 3)])
        let seed = int64(62)
        let result = mappingLayer.TranslateSeedForward(seed)
        Assert.AreEqual(int64(52), result)

    [<TestMethod>]
    member this.TestMappingLayerSeedsOutOfRange() =
        let mappingLayer = MappingLayer([Mapping(5, 15, 5); Mapping(50, 60, 3)])
        let seeds = [int64(1); int64(2)]
        let result = mappingLayer.TranslateSeedsForward(seeds)
        Assert.AreEqual(2, result.Length)
        Assert.AreEqual(int64(1), result.Item(0))
        Assert.AreEqual(int64(2), result.Item(1))

    [<TestMethod>]
    member this.TestMappingLayerSeeds1InRange() =
        let mappingLayer = MappingLayer([Mapping(5, 15, 5); Mapping(50, 60, 3)])
        let seeds = [int64(1); int64(15)]
        let result = mappingLayer.TranslateSeedsForward(seeds)
        Assert.AreEqual(2, result.Length)
        Assert.AreEqual(int64(1), result.Item(0))
        Assert.AreEqual(int64(5), result.Item(1))

    [<TestMethod>]
    member this.TestMappingLayerSeedsBothInRange() =
        let mappingLayer = MappingLayer([Mapping(5, 15, 5); Mapping(50, 60, 3)])
        let seeds = [int64(62); int64(5)]
        let result = mappingLayer.TranslateSeedsForward(seeds)
        Assert.AreEqual(2, result.Length)
        Assert.AreEqual(int64(52), result.Item(0))
        Assert.AreEqual(int64(5), result.Item(1))