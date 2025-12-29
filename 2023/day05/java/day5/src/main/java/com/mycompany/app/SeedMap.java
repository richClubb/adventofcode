package com.mycompany.app;

import java.util.Optional;

public class SeedMap {
    
    private long source_;
    private long target_;
    private long size_;

    public SeedMap(long source, long target, long size) {
        source_ = source;
        target_ = target;
        size_ = size;
    }

    public SeedMap(String seed_map_line) {
        String[] numbers = seed_map_line.split(" ");

        target_ = Long.parseLong(numbers[0]);
        source_ = Long.parseLong(numbers[1]);
        size_ = Long.parseLong(numbers[2]);

    }

    public Optional<Long> MapSeed(long value) {
        if( (value >= source_) && (value < (source_ + size_)))
        {
            return Optional.of(value - source_ + target_);
        }

        return Optional.empty();
    }
}
