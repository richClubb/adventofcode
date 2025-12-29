namespace tests

open System
open Microsoft.VisualStudio.TestTools.UnitTesting
open app_code

[<TestClass>]
type MappingTests () =

    [<TestMethod>]
    member this.TestMethodPassing () =
        Assert.IsTrue(true);

    [<TestMethod>]
    member this.TestMappingCreation () =
        let mapping = Mapping(5, 15, 5)
        Assert.AreEqual(int64(15), mapping.src_start)
        Assert.AreEqual(int64(19), mapping.src_end)
        Assert.AreEqual(int64(5), mapping.dest_start)
        Assert.AreEqual(int64(9), mapping.dest_end)

    [<TestMethod>]
    member this.TestTranslateSeedForwardOutOfRangeBelow() = 
        let mapping = Mapping(5, 15, 5)
        let seed = int64(2)
        let result = mapping.TranslateSeedForward(seed)
        Assert.AreEqual(int64(2), result)
    
    [<TestMethod>]
    member this.TestTranslateSeedForwardAtStartBoundary() = 
        let mapping = Mapping(5, 15, 5)
        let seed = int64(15)
        let result = mapping.TranslateSeedForward(seed)
        Assert.AreEqual(int64(5), result)

    [<TestMethod>]
    member this.TestTranslateSeedForwardInBoundary() = 
        let mapping = Mapping(5, 15, 5)
        let seed = int64(17)
        let result = mapping.TranslateSeedForward(seed)
        Assert.AreEqual(int64(7), result)

    [<TestMethod>]
    member this.TestTranslateSeedForwardAtEndBoundary() = 
        let mapping = Mapping(5, 15, 5)
        let seed = int64(19)
        let result = mapping.TranslateSeedForward(seed)
        Assert.AreEqual(int64(9), result)

    [<TestMethod>]
    member this.TestTranslateSeedForwardOutOfRangeAbove() = 
        let mapping = Mapping(5, 15, 5)
        let seed = int64(20)
        let result = mapping.TranslateSeedForward(seed)
        Assert.AreEqual(int64(20), result)

    [<TestMethod>]
    member this.TestTranslateSeedReverseOutOfRangeBelow() = 
        let mapping = Mapping(5, 15, 5)
        let seed = int64(2)
        let result = mapping.TranslateSeedReverse(seed)
        Assert.AreEqual(int64(2), result)
    
    [<TestMethod>]
    member this.TestTranslateSeedReverseAtStartBoundary() = 
        let mapping = Mapping(5, 15, 5)
        let seed = int64(5)
        let result = mapping.TranslateSeedReverse(seed)
        Assert.AreEqual(int64(15), result)

    [<TestMethod>]
    member this.TestTranslateSeedReverseInBoundary() = 
        let mapping = Mapping(5, 15, 5)
        let seed = int64(7)
        let result = mapping.TranslateSeedReverse(seed)
        Assert.AreEqual(int64(17), result)

    [<TestMethod>]
    member this.TestTranslateSeedReverseAtEndBoundary() = 
        let mapping = Mapping(5, 15, 5)
        let seed = int64(9)
        let result = mapping.TranslateSeedReverse(seed)
        Assert.AreEqual(int64(19), result)

    [<TestMethod>]
    member this.TestTranslateSeedReverseOutOfRangeAbove() = 
        let mapping = Mapping(5, 15, 5)
        let seed = int64(10)
        let result = mapping.TranslateSeedReverse(seed)
        Assert.AreEqual(int64(10), result)

    [<TestMethod>]
    member this.TestTranslateSeedForward1() = 
        let mapping = Mapping(52, 50, 98)
        let seed = int64(79)
        let result = mapping.TranslateSeedForward(seed)
        Assert.AreEqual(int64(81), result)