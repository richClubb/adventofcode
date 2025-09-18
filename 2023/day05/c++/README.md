std::optional had a 50% performance penalty

```cpp
for(SeedMap &seed_map : this->seed_maps)
{
    if (bool result; result = seed_map.map_seed(input))
    {
        return true;
    }
}
```

has a 10 second penalty over

```cpp
    for(uint64_t index = 0; index < this->seed_maps.size(); index++)
    {    
        if (bool result; result = this->seed_maps[index].map_seed(input))
        {
            return false;
        }
    }
```