using seed_map;

namespace seed_map_layer
{
    public class SeedMapLayer
    {
        List<SeedMap> seed_maps;

        public SeedMapLayer()
        {
            seed_maps = new List<SeedMap>();
        }

        public void AddSeedMap(SeedMap seed_map)
        {
            seed_maps.Add(seed_map);
        }

        public UInt64 MapSeed(UInt64 seed_value)
        {
            if (seed_maps.Count == 0)
            {
                return 0;
            }

            foreach (SeedMap seed_map in seed_maps)
            {
                UInt64? new_val = seed_map.MapSeed(seed_value);
                if ( new_val.HasValue )
                {
                    return new_val.Value;
                }
            }
            return seed_value;
        }
    }

    public class SeedMapLayers
    {
        List<SeedMapLayer> seed_map_layers;

        public SeedMapLayers()
        {
            seed_map_layers = new List<SeedMapLayer>();
        }

        public void AddSeedMapLayer(SeedMapLayer seed_map_layer)
        {
            seed_map_layers.Add(seed_map_layer);
        }

        public UInt64 MapSeed(UInt64 seed_value)
        {
            if (seed_map_layers.Count == 0)
            {
                return 0;
            }

            UInt64 temp_value = seed_value;
            foreach (SeedMapLayer seed_map_layer in seed_map_layers)
            {
                temp_value = seed_map_layer.MapSeed(temp_value);
            }
            return temp_value;
        }
    }
}