pylsh
===========

`pylsh` is a Python implementation of locality sensitive hashing with minhash. It is very useful for detecting near duplicate documents.

The implementation uses the MurmurHash v3 library to create document finger prints.

Cython is needed if you want to regenerate the `.cpp` files for the hashing and shingling code. By default the setup script uses the pregenerated `.cpp` sources, you can change this with the USE_CYTHON flag in setup.py

NumPy is needed to run the code.

The MurmurHash3 library is distributed under the MIT license. More information https://github.com/aappleby/smhasher


examples
============

For an overview of how LSH works and how to set the parameters see 
[this](http://nbviewer.jupyter.org/github/mattilyra/LSH/blob/master/examples/Introduction.ipynb)
notebook. The notebook is also available in the examples directory.

installation
============
```
> git clone https://github.com/mattilyra/LSH
> cd LSH
> python setup.py install
```
