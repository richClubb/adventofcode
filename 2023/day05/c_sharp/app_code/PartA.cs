using seed_map_layer;
using seed_map;

namespace part_a
{
    public class PartA
    {
        public static UInt64 Run(string input_file_path)
        {

            List<UInt64> seeds;
            SeedMapLayers seed_map_layers;

            (seeds, seed_map_layers) = injest_file(input_file_path);

            UInt64 min_value = UInt64.MaxValue;
            seeds.ForEach(seed => {
                    UInt64 new_value = seed_map_layers.MapSeed(seed);
                    if (min_value > new_value) min_value = new_value;
                }
            );

            return min_value;
        }

        private static (List<UInt64> seeds, SeedMapLayers seed_map_layers) injest_file(string input_file_path)
        {
            List<UInt64> seeds = new List<UInt64>();
            SeedMapLayers seed_map_layers = new SeedMapLayers();

            SeedMapLayer curr_seed_map_layer = new SeedMapLayer();

            foreach (var line in File.ReadLines(input_file_path))
            {
                if (line.Length == 0) continue;

                if (line.Contains("seeds: "))
                {
                    seeds = [.. line[7..].Split(" ").ToList().Select(x => UInt64.Parse(x))];
                    continue;
                }

                if (line.Contains(":"))
                {
                    curr_seed_map_layer = new SeedMapLayer();
                    seed_map_layers.AddSeedMapLayer(curr_seed_map_layer);
                    continue;    
                }

                curr_seed_map_layer.AddSeedMap(new SeedMap(line));
            }

            return (seeds, seed_map_layers);
        }
    }
}