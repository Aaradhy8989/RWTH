# RWTH
This work utilizes data-driven methods to morph a series of time-resolved experimental
OH-PLIF images into corresponding Temperature fields in the closed domain
of a premixed swirl combustor. The task is carried out with a fully convolutional network,
which is a type of convolutional neural network (CNN) used in many applications in machine
learning, alongside an existing experimental dataset which consists of simultaneous OH-PLIF
and PIV measurements in both attached and detached 
ame regimes. Two types of models
are compared: 1) a global CNN which is trained using images from the entire domain, and 2)
a set of local CNNs, which are trained only on individual sections of the domain.
