# wrapperPythonOptimPack

# Library needed

* OptimPack3 (Go to the branch v3devel if needed)
* pythonX-dev where X is 2 or 3
* pythonX-matplotlib
* pythonX-numpy
 
# How to

The goal is to first compile the C code into a dynamic library then calling this 
library in python.

`make all`

That is just calling `python3 setup.py build_ext --inplace`

Now in the folder should be **opkc_v3_1.cpython-35m-x86_64-linux-gnu.so** the 
extension depends of your python version.

Now just launch the Test.py or Test_3_1.py.

You may need to do:
`export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH`

# Note

This code was written with G.Heurtier and so the code *.{c,py} was written by 
both of us and the the code *3.1.{c,py} is the second version that should work 
with python3.

So the code that you should be used is the second version: *3.1.{c,py} because 
it use the latest version of OptimPack that is easier.

What you need in this case: Makefile, opkpy_v3.1.c, opkpy_v3_1.py and reading 
Test_3_1.py to understand how to use the code but it is not needed.
