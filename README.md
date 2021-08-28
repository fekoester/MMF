# MMF
Master Memory Function to calculate linear memory capacity of a time-delay based reservoir computer.
## Tutorial

You will need Armadillo to compile the program.

After compiling, start the program with ./nameOfProgram.

You will need to specifiy the parameters of the given reservoir system via the corresponding flags.

 *-a sets the local linearization parameter a.
 *-b sets the delay linearization parameter b.
 *-theta sets the virtual node distance interval time.
 *-N sets the number of virtual nodes.
 *-tauInTheta sets the delay length in number of theta intervals.
 *-regularisationParameter sets the tikhonov regularisation parameter.
 *-regularisationParameterRelative is a boolean, setting the tikhonov regularisation parameter relative to the highest value in the state matrix.
