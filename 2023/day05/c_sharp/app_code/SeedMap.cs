

namespace seed_map
{
    public class SeedMap
    {
        UInt64 source_;
        UInt64 target_;
        UInt64 size_;

        public UInt64 Source { get=> source_;}
        public UInt64 Target { get=> target_;}
        public UInt64 Size { get=> size_;}

        public SeedMap(UInt64 source, UInt64 target, UInt64 size)
        {
            source_ = source;
            target_ = target;
            size_ = size;
        }

        public SeedMap(string seed_map_string)
        {
            var seed_strings = seed_map_string.Split(" ");
            if (seed_strings.Length != 3)
            {
                throw new Exception("Can't parse the line");
            }

            source_ = UInt64.Parse(seed_strings[1]);
            target_ = UInt64.Parse(seed_strings[0]);
            size_ = UInt64.Parse(seed_strings[2]);
        }

        public UInt64? MapSeed(UInt64 seed)
        {
            if (
                (source_ <= seed) && 
                ((source_ + size_) > seed)
            )
            {
                return seed - source_ + target_;
            }

            return null;
        }
    }
}