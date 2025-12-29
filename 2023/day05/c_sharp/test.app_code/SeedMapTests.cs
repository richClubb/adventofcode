using seed_map;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace test_seed_map;

[TestClass]
public class SeedMapTests
{
    [TestMethod]
    [DataRow(1UL, null)]
    [DataRow(2UL, 3UL)]
    [DataRow(3UL, 4UL)]
    [DataRow(4UL, 5UL)]
    [DataRow(5UL, 6UL)]
    [DataRow(6UL, null)]
    
    public void TestMapSeed(UInt64 input, UInt64? expected)
    {
        SeedMap seed_map = new SeedMap(2, 3, 4);

        Assert.AreEqual(expected, seed_map.MapSeed(input));
    }


    [TestMethod]
    [DataRow("1 2 3", 1U, 2U, 3U)]
    public void TestSeedMapStringConstructor(string initial, UInt64 expected_target, UInt64 expected_source, UInt64 expected_size)
    {
        SeedMap seed_map = new SeedMap(initial);

        Assert.AreEqual(expected_source, seed_map.Source);
        Assert.AreEqual(expected_target, seed_map.Target);
        Assert.AreEqual(expected_size, seed_map.Size);
    }
}