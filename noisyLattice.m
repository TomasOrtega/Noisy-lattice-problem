clear; clc;
side = 5;
n = side*side; % has to be a perfect square cause I'm lazy
trueV1 = [0; 1];
trueV2 = [1; 0];
noiseSigma = 0.05;
%% Initialize lattice
startLattice = zeros(2, n);
z = 1;
for i = 1:side
    for j = 1:side
        startLattice(:,z) = [i; j];
        z = z + 1;
    end
end
startLattice = startLattice + randn(2, n) * noiseSigma;
x = startLattice(1, :);
y = startLattice(2, :);
scatter(x, y);

%% Now we will do successive optimization
or = startLattice(:, 2); % I choose it deterministically cause I'm lazy, you will want to do it different
v1 = [0.01; 0.99];
v2 = [0.99; -0.01];
latticeCoords = zeros(2, n);

nIterations = 20; % I'm lazy and I do not want to make a stopping criteria
for it = 1:nIterations
    % Start optimizing the coordinates
    for z = 1:n
        % Again, I'm lazy, this is quadratic and can be done constant
        % I impose coordinates must be between -n and n
        distInit = norm(or + latticeCoords(1, z)*v1 + latticeCoords(2, z)*v2 - startLattice(:, z));
        for i = -n:n
            for j = -n:n
                distAux = norm(or + i*v1 + j*v2 - startLattice(:, z));
                if distAux < distInit
                    distInit = distAux;
                    latticeCoords(:, z) = [i; j];
                end
            end
        end
    end

    % Then we optimize vectors and origin
    x0 = [or(1), or(2), v1(1), v1(2), v2(1), v2(2)];
    fCost = @(x) cost(x, n, latticeCoords, startLattice);
    x = fminsearch(fCost, x0);
    or = [x(1); x(2)];
    v1 = [x(3); x(4)];
    v2 = [x(5); x(6)];
    plotLattice(startLattice, latticeCoords, or, v1, v2);
end

function res = plotLattice(startLattice, coords, or, v1, v2)
    % coords is a 2 x n array, n >= 2
    % or is the origin of coordinates
    % v1 and v2 are column vectors, the basis of the lattice
    x = startLattice(1, :);
    y = startLattice(2, :);
    hold on;
    scatter(x, y);
    points = or + coords(1, :) .* v1 + coords(2, :) .* v2;
    x = points(1, :);
    y = points(2, :);
    scatter(x, y, 'x');
    hold off;
end

function res = cost(x, n, latticeCoords, startLattice)
    res = 0;
    or = [x(1); x(2)];
    v1 = [x(3); x(4)];
    v2 = [x(5); x(6)];
    for z = 1:n
        res = res + norm(or + latticeCoords(1, z)*v1 + latticeCoords(2, z)*v2 - startLattice(:, z))^2;
    end
end