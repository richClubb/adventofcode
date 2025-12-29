C using external lib
--------------------

# Build / Run

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ../
make
```

I've had a problem with it not working on Debug builds so at the moment it only works for "Release" candidates.

```
./day5 -i [path] -r [run]
```

Where [run] is:
* part_a
* part_b