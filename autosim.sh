#!/bin/bash

for x in `seq 0 1`; do
  echo "Running file $x"
  filename = "geometry/test/test_cylinder_{x}.in"
  python -m gprMax "geometry/test/test_cylinder_$x.in" -n 43
done

