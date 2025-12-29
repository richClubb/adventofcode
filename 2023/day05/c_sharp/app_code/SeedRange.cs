using System.Diagnostics;
using seed_map_layer;


namespace seed_range
{


    public class SeedRange
    {
        UInt64 start_;
        UInt64 size_;

        public UInt64 Start {
            get => start_;
        }

        public UInt64 End
        {
            get => start_ + size_;
        }

        public UInt64 Size
        {
            get => size_;
        }

        public SeedRange(UInt64 start, UInt64 size)
        {
            start_ = start;
            size_ = size;
        }
    }
}