Advent of code 2023 Day 05 - Java
---------------------------------

# Overview

This was fine, the paradigm is the same as C# so there isn't much to say...

Except.

Java as an optional type ("Yay") but you can't use primitive types so boo. This didn't waste me an embarasing amount of time.

```Java
Optional<long> thing;
```
Is incorrect

```Java
Optional<Long> thing;
```

This seems to be a weird thing to do.

I also chose `maven` to do the package management and I don't really like it very much. Probably an education thing but as a newbie it wasn't particularly friendly. This is probably a skill issue and a lack of time spent learning.

# Build / Run

```
mvn package
mvn clean compile assembly:single
java -jar target/day5-1.0-SNAPSHOT-jar-with-dependencies.jar --file ../../full_data.txt --run part_b
```

# Test 

Haven't done any unit tests for this.