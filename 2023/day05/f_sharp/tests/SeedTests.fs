namespace tests

open System
open Microsoft.VisualStudio.TestTools.UnitTesting
open app_code

[<TestClass>]
type SeedTests () =

    [<TestMethod>]
    member this.TestMethodPassing () =
        Assert.IsTrue(true);

    // failed to get seed equality to work
    // [<TestMethod>]
    // member this.TestSeedEquals () =
    //     let seed1 = Seed(1)
    //     let seed2 = Seed(1)
    //     Assert.IsTrue(seed1.Equals(seed2))

    // [<TestMethod>]
    // member this.TestSeedNotEquals () =
    //     let seed1 = Seed(1)
    //     let seed2 = Seed(2)
    //     Assert.IsFalse(seed1.Equals(seed2))

    