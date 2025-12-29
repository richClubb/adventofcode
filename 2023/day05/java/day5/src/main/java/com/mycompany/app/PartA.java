package com.mycompany.app;

import java.io.File;
import java.util.ArrayList;
import java.util.Scanner;

public class PartA {
    
    public static long Run(String file_path)
    {
        ArrayList<Object> injest_result = injest_file(file_path);

        ArrayList<Long> seeds = (ArrayList<Long>)(injest_result.get(0));
        SeedMapLayers seed_map_layers = (SeedMapLayers)(injest_result.get(1));

        long min_value = Long.MAX_VALUE;
        for (long seed : seeds)
        {
            long result = seed_map_layers.MapSeed(seed);
            if (result < min_value) min_value = result;
        }

        return min_value;
    }

    private static ArrayList<Object> injest_file(String file_path)
    {
        ArrayList<Object> results = new ArrayList<Object>(2);

        Scanner scanner = null;
        try {
            scanner = new Scanner(new File(file_path));
        }
        catch (Exception ex)
        {
            new Exception("");
        }

        SeedMapLayer curr_layer = null;
        SeedMapLayers seed_map_layers = new SeedMapLayers();

        while(scanner.hasNextLine())
        {
            String line = scanner.nextLine();
        
            if (line.length() == 0) continue;

            if (line.contains("seeds: "))
            {
                results.add(ExtractSeeds(line.substring(7)));
                continue;
            }

            if (line.contains(":"))
            {
                curr_layer = new SeedMapLayer();
                seed_map_layers.AddSeedMapLayer(curr_layer);
                continue;
            }

            curr_layer.AddSeedMap(new SeedMap(line));
        }

        results.add(seed_map_layers);

        return results;
    }

    private static ArrayList<Long> ExtractSeeds(String seeds_string)
    {
        ArrayList<Long> result = new ArrayList<>();
        String[] seeds_strings = seeds_string.split(" ");

        for(String seed_string : seeds_strings)
        {
            Long value = Long.parseLong(seed_string);
            result.add(value);
        }

        return result;
    }
}
