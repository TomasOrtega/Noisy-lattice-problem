# Noisy-lattice-problem

The problem: we have a picture of a lattice, where the points have some Gaussian noise on top, and we want to obtain the de-noised lattice points.

See the image below for an example. The circles are the given noisy measurements, which we are given initially.
We then compute the minimum square error lattice, plotted with dotted lines in the figure. The crosses are where the adjusted measurements fall in the lattice.

![Lattice Example](example.png)

An iterative procedure is used to obtain the solution, seen in the animation below.

![Lattice Animation](animated.gif)
