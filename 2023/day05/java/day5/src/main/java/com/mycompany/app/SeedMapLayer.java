package com.mycompany.app;

import java.util.ArrayList;
import java.util.Optional;

public class SeedMapLayer {
    
   private ArrayList<SeedMap> seed_maps;

    public SeedMapLayer()
    {
        seed_maps = new ArrayList<SeedMap>();
    }

    public void AddSeedMap(SeedMap seed_map)
    {
        seed_maps.add(seed_map);
    }

    public long MapSeed(long seed_value)
    {
        for (SeedMap seed_map : seed_maps) {
            Optional<Long> result = seed_map.MapSeed(seed_value);

            if (result.isPresent()) return result.get(); 
        }

        return seed_value;
    }
}
