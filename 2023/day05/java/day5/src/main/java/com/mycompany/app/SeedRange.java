package com.mycompany.app;

public class SeedRange {
    
    private long start_;
    private long size_;
    private long end_;

    public SeedRange(long start, long size) {
        start_ = start;
        size_ = size;
        end_ = start + size;
    }

    public Long getStart() { return start_; };
    public Long getEnd() { return end_; };
    public Long getSize() { return size_; };
}
