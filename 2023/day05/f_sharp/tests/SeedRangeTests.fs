namespace tests

open System
open Microsoft.VisualStudio.TestTools.UnitTesting
open app_code

[<TestClass>]
type SeedRangeTests () =

    [<TestMethod>]
    member this.TestMethodPassing () =
        Assert.IsTrue(true);

    [<TestMethod>]
    member this.TestSeedTranslation () = 
        let seedRange = SeedRange(int64(1), int64(10))
        let mappingLayers = [MappingLayer([|Mapping(5, 50, 5); Mapping(15, 1, 5)|]);MappingLayer([|Mapping(5, 50, 5); Mapping(35, 6, 5)|])]
        let result = seedRange.FindMinSeedInRange(mappingLayers)
        Assert.AreEqual(int64(15), result)

    [<TestMethod>]
    member this.TestSeedTranslationMutable () = 
        let seedRange = SeedRange(int64(1), int64(10))
        let mappingLayers = [MappingLayer([|Mapping(5, 50, 5); Mapping(15, 1, 5)|]);MappingLayer([|Mapping(5, 50, 5); Mapping(35, 6, 5)|])]
        let result = seedRange.FindMinSeedInRangeMutable(mappingLayers)
        Assert.AreEqual(int64(15), result)