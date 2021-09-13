# MMF
Master Memory Function to calculate linear memory capacity of a time-delay based reservoir computer.
## Tutorial

You will need Armadillo and Boost to compile the program. 
Armadillo is a C++ linear algebra package, which can be found here (http://arma.sourceforge.net/).
Boost is a C++ library, which can be found here (https://www.boost.org/).
Otherwise the program uses basic C++ libraries, thus it should compile on every system.

After compiling, start the program with ./nameOfProgram.

You will need to specifiy the parameters of the given reservoir system via the corresponding flags.

 * "-a" sets the local linearization parameter a.
 * "-b" sets the delay linearization parameter b.
 * "-theta" sets the virtual node distance interval time.
 * "-N" sets the number of virtual nodes.
 * "-tauInTheta" sets the delay length in number of theta intervals.
 * "-regularisationParameter" sets the tikhonov regularisation parameter.
 * "-regularisationParameterRelative" is a boolean, setting the tikhonov regularisation parameter relative to the highest value in the state matrix.
 * "-filename" takes a string as name for the file, to which the output is then written.

One example call of the program would be "./MMF -a -0.5 -b 0.4 -N 100 -tauInTheta 141 -theta 0.5"

## ArXiv Paper

https://arxiv.org/abs/2108.12643
