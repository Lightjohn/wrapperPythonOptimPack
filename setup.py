from distutils.core import setup, Extension
import numpy as np

# One way       ------------------------------
# module1 = Extension("opk_get_reason", ["OptimPackModule.c", "/obs/gheurtier/OptimPack/src/error.c"])
# module2 = Extension("opk_guess_status", ["OptimPackModule.c", "/obs/gheurtier/OptimPack/src/error.c"])
# module5 = Extension("opk_get_object_references", ["OptimPackModule.c", "/obs/gheurtier/OptimPack/src/algebra.c"])


# An other...   ------------------------------
# module6 = Extension("opk_drop_object", ["OptimPackModule.c"])


# An other...   ------------------------------
optimpack = Extension("opkc_v3", sources=["opkc_v3.c"], libraries=["opk"], library_dirs=["/obs/gheurtier/lib"])

# The setup     ------------------------------
setup(name="OptimPack", version="1.0", description="Python Package of OptimPack", author="Eric Thiebaut",
      include_dirs=[np.get_include()], ext_modules=[optimpack])
