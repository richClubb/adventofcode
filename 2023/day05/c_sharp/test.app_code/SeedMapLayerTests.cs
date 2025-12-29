using seed_map;
using seed_map_layer;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace test.seed_map_layer;

[TestClass]
public class SeedMapLayerTests
{
    [TestMethod]
    [DataRow(1U, 1U)]
    [DataRow(2U, 3U)]
    [DataRow(3U, 4U)]
    [DataRow(4U, 5U)]
    [DataRow(5U, 6U)]
    [DataRow(6U, 6U)]
    [DataRow(19U, 19U)]
    [DataRow(20U, 30U)]
    [DataRow(21U, 31U)]
    [DataRow(22U, 32U)]
    [DataRow(23U, 33U)]
    [DataRow(24U, 24U)]
    public void TestMapSeed(UInt64 input, UInt64 expected)
    {
        SeedMapLayer seed_map_layer = new SeedMapLayer();
        
        {
            SeedMap seed_map = new SeedMap(2, 3, 4);
            seed_map_layer.AddSeedMap(seed_map);
        
        }
        {
            SeedMap seed_map = new SeedMap(20, 30, 4);
            seed_map_layer.AddSeedMap(seed_map);
        }

        Assert.AreEqual(expected, seed_map_layer.MapSeed(input));
    }
}