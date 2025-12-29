C lib src
---------

# Build / Run

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ../
make
``

You can then manually copy the output into the `lib/release` directory.

I've had a problem with debug builds not creating the right symbols so it only works for "Release" build.