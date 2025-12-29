using seed_map;
using seed_map_layer;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace test.seed_map_layers;

[TestClass]
public class SeedMapLayersTests
{
    [TestMethod]
    [DataRow(1U, 1U)]
    [DataRow(2U, 3U)]
    [DataRow(3U, 4U)]
    [DataRow(4U, 5U)]
    [DataRow(5U, 6U)]
    [DataRow(6U, 6U)]
    [DataRow(19U, 19U)]
    [DataRow(20U, 40U)]
    [DataRow(21U, 41U)]
    [DataRow(22U, 42U)]
    [DataRow(23U, 43U)]
    [DataRow(24U, 24U)]
    public void TestMapSeed(UInt64 input, UInt64 expected)
    {
        SeedMapLayers seed_map_layers = new SeedMapLayers();
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
            seed_map_layers.AddSeedMapLayer(seed_map_layer);
        }

        {
            SeedMapLayer seed_map_layer = new SeedMapLayer();
        
            {
                SeedMap seed_map = new SeedMap(10, 15, 4);
                seed_map_layer.AddSeedMap(seed_map);
            }
            {
                SeedMap seed_map = new SeedMap(30, 40, 4);
                seed_map_layer.AddSeedMap(seed_map);
            }
            seed_map_layers.AddSeedMapLayer(seed_map_layer);
        }

        Assert.AreEqual(expected, seed_map_layers.MapSeed(input));
    }
}