import numpy as np
from scipy.optimize import fmin
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt


def denoise_lattice(noisy_mes, show_plot=True):
    """
    Computes the minimum Mean Square Error (MSE) lattice given the input noisy measurements.

    Args:
        noisy_mes: A 2 x n measurement vector, where n is >= 2.
        show_plot: Whether to display the plot (default: True)

    Returns:
        coords: Integer coordinates in lattice space
        or: Origin of the lattice
        v1, v2: Basis vectors of the denoised lattice
    """
    # Input validation
    if not np.issubdtype(noisy_mes.dtype, np.number):
        raise TypeError("Input must contain numeric values")
    if noisy_mes.ndim != 2 or noisy_mes.shape[0] != 2:
        raise ValueError("Input must be a 2×n array")
    if noisy_mes.shape[1] < 4:
        raise ValueError("At least 4 points are required")

    # Plot noisy data
    n = noisy_mes.shape[1]
    if show_plot:
        plt.figure(figsize=(8, 8))
        plt.scatter(noisy_mes[0], noisy_mes[1], c="g", linewidth=4)

    # Pick an origin and initial basis
    p_dist = squareform(pdist(noisy_mes.T))
    g = np.argmin(
        np.sum(p_dist, axis=0)
    )  # pick the origin as the point with min sum distance
    idx = np.argsort(p_dist[:, g])  # indexes of points from nearest to furthest of g
    or_ = noisy_mes[:, g]
    v1 = noisy_mes[:, idx[1]] - or_  # Vector from g to its closest neighbor
    v2 = np.array([-v1[1], v1[0]])  # Perpendicular of previous vector

    # Successive optimization
    coords = np.zeros((2, n))

    for it in range(4, n + 1):
        print(f"Iteration number {it}")
        # Start optimizing the coordinates
        for z in range(it):
            dist_init = np.linalg.norm(
                or_
                + coords[0, idx[z]] * v1
                + coords[1, idx[z]] * v2
                - noisy_mes[:, idx[z]]
            )

            # The true integer coordinates have to be in a +-1 distance from
            # the floating point ones
            lambdas = np.zeros(2)

            if np.linalg.norm(noisy_mes[:, idx[z]] - or_) > 1e-3:
                lambdas = np.linalg.solve(
                    np.column_stack((v1, v2)), noisy_mes[:, idx[z]] - or_
                )

            # I impose coordinates must be between -it and it
            int_lambdas = np.round(lambdas)
            int_lambdas[int_lambdas > it] = it
            int_lambdas[int_lambdas < -it] = -it

            for i in range(-1, 2):
                for j in range(-1, 2):
                    aux_coords = np.array([i, j]) + int_lambdas
                    dist_aux = np.linalg.norm(
                        or_
                        + np.dot(np.column_stack((v1, v2)), aux_coords)
                        - noisy_mes[:, idx[z]]
                    )

                    if dist_aux < dist_init:
                        dist_init = dist_aux
                        coords[:, idx[z]] = aux_coords

        # Then we optimize vectors and origin
        x0 = np.array([or_[0], or_[1], v1[0], v1[1], v2[0], v2[1]])
        x, total_sq_err = fmin(
            cost,
            x0,
            args=(it, coords[:, idx[:it]], noisy_mes[:, idx[:it]]),
            full_output=True,
            maxiter=1000,  # Increase max iterations
            maxfun=2000,  # Increase max function evaluations
            ftol=1e-8,  # Tighter function tolerance
            xtol=1e-8,  # Tighter parameter tolerance
        )[:2]
        print(f"Total square error: {total_sq_err}")

        or_ = np.array([x[0], x[1]])
        v1 = np.array([x[2], x[3]])
        v2 = np.array([x[4], x[5]])
        if show_plot:
            plot_lattice(noisy_mes[:, idx[:it]], coords[:, idx[:it]], or_, v1, v2)

    # Plot the final result
    if show_plot:
        plot_lattice(noisy_mes, coords, or_, v1, v2)
        plt.show()

    return coords, or_, v1, v2


def plot_grid(coords, or_, v1, v2):
    """Helper function to plot the lattice grid"""
    minx = np.min(coords[0])
    maxx = np.max(coords[0])
    miny = np.min(coords[1])
    maxy = np.max(coords[1])

    x, y = np.meshgrid(np.arange(minx, maxx + 1), np.arange(miny, maxy + 1))
    xy = np.column_stack((x.flatten(), y.flatten()))
    T = np.vstack((v1, v2))
    xyt = np.dot(xy, T)
    xt = xyt[:, 0].reshape(x.shape)
    yt = xyt[:, 1].reshape(y.shape)

    plt.plot(xt + or_[0], yt + or_[1], "r:", linewidth=0.1)
    plt.plot(xt.T + or_[0], yt.T + or_[1], "r:", linewidth=0.1)


def plot_lattice(noisy_mes, coords, or_, v1, v2):
    """Helper function to plot the lattice points and grid"""
    plt.clf()
    plt.scatter(noisy_mes[0], noisy_mes[1], c="g", linewidth=4)

    # Reshape vectors for proper broadcasting
    v1_reshaped = v1.reshape(2, 1)
    v2_reshaped = v2.reshape(2, 1)

    # Calculate points using matrix multiplication
    points = (
        or_.reshape(2, 1)
        + np.dot(v1_reshaped, coords[0].reshape(1, -1))
        + np.dot(v2_reshaped, coords[1].reshape(1, -1))
    )

    plt.scatter(points[0], points[1], marker="x", c="r", linewidth=2)
    plot_grid(coords, or_, v1, v2)
    plt.axis("square")


def cost(x, n, coords, noisy_mes):
    """Cost function for optimization"""
    or_ = np.array([x[0], x[1]])
    v1 = np.array([x[2], x[3]])
    v2 = np.array([x[4], x[5]])

    # Reshape vectors for proper broadcasting
    v1_reshaped = v1.reshape(2, 1)
    v2_reshaped = v2.reshape(2, 1)

    # Calculate points using matrix multiplication
    points = (
        or_.reshape(2, 1)
        + np.dot(v1_reshaped, coords[0].reshape(1, -1))
        + np.dot(v2_reshaped, coords[1].reshape(1, -1))
    )

    # Calculate squared error
    res = np.sum(np.sum((points - noisy_mes) ** 2, axis=0))
    return res


if __name__ == "__main__":
    # Example usage
    # Generate some random noisy lattice points
    np.random.seed(42)
    n_points = 20
    true_v1 = np.array([1.0, 0.2])
    true_v2 = np.array([-0.2, 1.0])
    true_or = np.array([0.0, 0.0])

    # Generate integer coordinates
    coords = np.random.randint(-5, 6, (2, n_points))

    # Generate noisy measurements
    v1_reshaped = true_v1.reshape(2, 1)
    v2_reshaped = true_v2.reshape(2, 1)

    noisy_points = (
        true_or.reshape(2, 1)
        + np.dot(v1_reshaped, coords[0].reshape(1, -1))
        + np.dot(v2_reshaped, coords[1].reshape(1, -1))
    )
    noisy_points += np.random.normal(0, 0.1, noisy_points.shape)

    # Run the denoising algorithm
    recovered_coords, recovered_or, recovered_v1, recovered_v2 = denoise_lattice(
        noisy_points
    )
