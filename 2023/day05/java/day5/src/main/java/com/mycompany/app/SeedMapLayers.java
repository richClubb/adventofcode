package com.mycompany.app;

import java.util.ArrayList;

public class SeedMapLayers {
    
    private ArrayList<SeedMapLayer> seed_map_layers;

    public SeedMapLayers()
    {
        seed_map_layers = new ArrayList<SeedMapLayer>();
    }

    public void AddSeedMapLayer(SeedMapLayer seed_map_layer)
    {
        seed_map_layers.add(seed_map_layer);
    }

    public long MapSeed(long seed_value)
    {
        long result = seed_value;
        for (SeedMapLayer seed_map_layer: seed_map_layers)
        {
            result = seed_map_layer.MapSeed(result);
        }

        return result;
    }
}
