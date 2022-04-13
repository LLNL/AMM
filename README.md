## AMM, Version 0.1

**Adaptive Multilinear Meshes (AMMs)** is a new framework to represent piecewise
multilinear volumetric data using mixed-precision adaptive meshes. AMM is
designed to reduce in-memory and on-disk data footprint using a spatial
hierarchy with "rectangular cuboidal" cells. AMM also supports mixed-precision
representation of function values in byte-sized increments. AMM can be
constructed through arbitrary data streams. The current version provides several
examples of data streams that can be selected through command line.

For details on AMM's data structure and representation, please consult the
[publication](https://ieeexplore.ieee.org/document/9751449).

#### Installing AMM

##### Dependencies

* AMM requires a `C++` compiler that supports `C++ 17` (tested with `GNU gcc 7.3.0`).
* AMM uses the [Visualization Toolkit (VTK)](https://www.vtk.org/ "VTK") to
output AMM meshes (developed with `v 8.1.2`).


##### Installation

To download and install AMM, please use the following steps.
```
$ git clone git@github.com:LLNL/AMM.git
$ cd AMM
$ mkdir build; cd build
$ cmake ../
$ make -j12
```


##### Customizations to AMM build

By default, all function values are stored at full precision (same as precision
  of the input data, `float` or `double`). To enable mixed-precision representation,
  use
```
$ cmake -DAMM_ENABLE_PRECISION ../
```

**Note**: The file `<path_to_clone>/amm/macros.hpp` defines several macros that
may be used to further customize the functionality of AMM. For example,
the `MAX_DEPTH` of the AMM representation may be reduced if needed.


#### Using AMM

APP provides a simple command-line interface in the `app` directory. The
application `amm-cli` provides options to read a binary file that defines an input function, and creates a vtk unstructured grid output file
(`*.vtu`).

There are three modes of inputs to `amm-cli`.
* `func`: a binary file with `XxYxZ` values of type `u8`, `f32`, or `f64`. Currently, AMM supports `f32` and `f64` (`u8` inputs are cast to `f32`). In this mode, the tool first computes the wavelet coefficients, then computes the stream, and finally passes the stream to AMM creation pipeline.
* `wcoeffs`: pre-computed wavelet coefficients can be passed directly to the tool as well.
  - Expected precision: `f32` or `f64`.
  - Expected size:  `[2^L+1 x 2^L+1 x 2^L+1]`. Note that AMM internally stores a spatial hierarchy that is `power of two plus 1`. Refer the paper to note why this is needed and how its effects are mitigated. Nevertheless, currently, the wavelet coefficients are expected to be given in this expanded domain.
* `amm`: a pre-computed AMM mesh (stored as`*.vtu` files) can be loaded. Currently, the loading functionality works correctly. This is useful to expand upon and testing iteration capabilities in AMM and improvements in internal representation.


For detailed information, see the command line options,
```
$ ./amm-cli

Usage: ./amm-cli <args>
	 --input  [file_name] [func/wcoeffs/amm] [dtype]: input filename, type of input, precision of input

                               func:                      input function read from raw binary file of specific datatype (u8, f32, f64)
                               wcoeffs:                   wavelet coefficients read from raw binary file of specific datatype (f32, f64)
                               amm:                       AMM read from a vtk unstructued mesh (precision not specified)

                               dtype:                     u8/f32/f64

	 --dims   [X] [Y] [Z=1]:                              dimensions of data [default: Z = 1])

	 --stream [stream_type]:                              type of stream to create
                               1:                                   by row major
                               2:                                   by subband row major
                               3:                                   by coeff wavelet norm
                               4:                                   by wavelet norm
                               5:                                   by level
                               6:                                   by bitplane
                               7:                                   by magnitude
	 --chunk  [chunk_size=0] [count/kb]:                  size of each chunks, either as number of coefficients or amount of data (in kb) [default: chunk_size = full stream]
	 --end    [end_val=0] [count/kb/val]:                 end of the stream, either as number of coefficients, amount of data (in kb), or threshold value of the coefficients [default: end_val = full stream]

	 --rect   [0/1 = 1]:                                  enable rectangular nodes in AMM [default: true]
	 --improp [0/1 = 0]:                                  enable improper node in AMM [default: false]
	 --wvlts  [approx/interp = approx]:                   use approximating or interpolating wavelet basis [default: approximating]
	 --extrap [zero/linear/linlift = linlift]:            extrapolate data using zero-padding, linear extrapolation, or linear-lifting method [default: linear-lifting]
	 --wnorm  [0/1 = 0]:                                  normalize wavelet basis [default: false]
	 --wdepth [depth = 1]:                                compute wavelet transform up until the specified depth [default: depth=1]

	 --novalidate                                         do not validate the output
	 --outpath [path]                                     path for the output vtk file [default: no output]
	 --lowres                                             also write the lowres function (same path as above)
	 --wavout [file_name]                                 output the wavelet coefficients to a file
```


#### Credits

AMM has been developed by [Harsh Bhatia](http://www.sci.utah.edu/~hbhatia) (hbhatia@llnl.gov), with contributions from [Duong Hoang](https://hoang-dt.github.io/). Other team members with valuable inputs on design include Peer-Timo Bremer, Valerio Pascucci, and Peter Lindstrom.



#### License

AMM is released under the terms of BSD-3 license. See the `LICENSE` file for details.

*`LLNL-CODE-777060`*
