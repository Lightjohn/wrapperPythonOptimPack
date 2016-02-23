# Gathering of all the executable to be made
all: opk

# Creation of the executable, all objects that are meant to be created or used are written
opk:
	python setup.py build_ext --inplace -I/obs/gheurtier/anaconda/lib -L/usr/lib/python2.7

# build_ext is to build extension module,  --inplace will put the .so in the current dir.

opk3:
	python3 setup.py build_ext --inplace
# Delete all the transitional files
clean:
	rm *.o

# Delete all that can be regenerated and allow a complete reconstruction of the project
# mrproper: clean
#	rm -rf MonExecutable

# -I/usr/include/python2.7
				
