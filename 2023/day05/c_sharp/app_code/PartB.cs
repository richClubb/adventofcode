using seed_map;
using seed_map_layer;
using seed_range;

namespace part_b
{
    public class PartB
    {
        public static UInt64 Run(string input_file_path)
        {

            List<SeedRange>  seed_ranges;
            SeedMapLayers seed_map_layers;

            (seed_ranges , seed_map_layers) = injest_file(input_file_path);

            UInt64 min_value = UInt64.MaxValue;
            seed_ranges.ForEach(seed_range => {
                    UInt64 range_min = UInt64.MaxValue;
                    for(UInt64 seed = seed_range.Start; seed < seed_range.End; seed++)
                    {
                        UInt64 new_value = seed_map_layers.MapSeed(seed);
                        if (range_min > new_value) range_min = new_value;
                    }
                    
                    if (range_min < min_value) min_value = range_min;
                }
            );

            return min_value;
        }

        private static List<SeedRange> extract_seed_ranges(string numbers)
        {
            List<SeedRange> seed_ranges = new List<SeedRange>();
            string[] numbers_strings = numbers.Split(" ");
            for(int index = 0; index < numbers_strings.Length; index+=2)
            {
                seed_ranges.Add(new SeedRange(UInt64.Parse(numbers_strings[index]), UInt64.Parse(numbers_strings[index + 1])));
            }

            return seed_ranges;
        }

        private static (List<SeedRange> seed_ranges, SeedMapLayers seed_map_layers) injest_file(string input_file_path)
        {
            List<SeedRange> seed_ranges = new List<SeedRange>();
            SeedMapLayers seed_map_layers = new SeedMapLayers();

            SeedMapLayer curr_seed_map_layer = new SeedMapLayer();

            foreach (var line in File.ReadLines(input_file_path))
            {
                if (line.Length == 0) continue;

                if (line.Contains("seeds: "))
                {
                    seed_ranges = extract_seed_ranges(line[7..]);
                    continue;
                }

                if (line.Contains(':'))
                {
                    curr_seed_map_layer = new SeedMapLayer();
                    seed_map_layers.AddSeedMapLayer(curr_seed_map_layer);
                    continue;    
                }

                curr_seed_map_layer.AddSeedMap(new SeedMap(line));
            }

            return (seed_ranges, seed_map_layers);
        }
    }
}