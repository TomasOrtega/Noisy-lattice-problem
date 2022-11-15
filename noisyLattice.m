clear; clc; close all;
format long;

%% Generate noisy measurements
side = 10;
n = side*side; % has to be a perfect square cause I'm lazy
trueV1 = [1; 11];
trueV2 = [11; -1];
noiseSigma = 0.3;
noisyMes = zeros(2, n); % noisy measurements
z = 1;
for i = 1:side
    for j = 1:side
        noisyMes(:,z) = [trueV1, trueV2] * [i; j];
        z = z + 1;
    end
end
noisyMes = noisyMes + randn(2, n) * noiseSigma;
x = noisyMes(1, :);
y = noisyMes(2, :);
scatter(x, y,'linewidth', 4,'MarkerEdgeColor', 'g');

%% Create good initial guess
or = noisyMes(:, 20); % Maybe you will want to do it different
diffs = noisyMes(:, 2:n) - noisyMes(:, 1:n-1);
v1 = [median(diffs(1)); median(diffs(2))]; % You want to do something better than this
v2 = [-v1(2); v1(1)];

%% Successive optimization
coords = zeros(2, n);
nIterations = 3; % I'm lazy and I do not want to make a stopping criteria
for it = 1:nIterations
    disp("Iteration number " + it)
    % Start optimizing the coordinates
    for z = 1:n
        distInit = norm(or + coords(1, z)*v1 + coords(2, z)*v2 - noisyMes(:, z));
        % The true integer coordinates have to be in a +-1 distance from
        % the floating point ones
        lambdas = [0; 0];
        if (norm(noisyMes(:, z) - or) > 1e-3)
            lambdas = linsolve([v1, v2], noisyMes(:, z) - or);
        end
        % I impose coordinates must be between -n and n
        intLambdas = round(lambdas);
        intLambdas(intLambdas > n) = n;
        intLambdas(intLambdas < -n) = -n;
        for i = -2:2
            for j = -2:2
                auxCoords = [i; j] + intLambdas;
                distAux = norm(or + [v1, v2] * auxCoords - noisyMes(:, z));
                if distAux < distInit
                    distInit = distAux;
                    coords(:, z) = auxCoords;
                end
            end
        end
    end

    % Then we optimize vectors and origin
    x0 = [or(1), or(2), v1(1), v1(2), v2(1), v2(2)];
    fCost = @(x) cost(x, n, coords, noisyMes);
    [x, totalSqErr] = fminsearch(fCost, x0);
    disp("Total square error: " + totalSqErr)
    or = [x(1); x(2)];
    v1 = [x(3); x(4)];
    v2 = [x(5); x(6)];
    % if it == nIterations
    %   plotLattice(startLattice, latticeCoords, or, v1, v2);
    % end
end

%% Plot the result
plotLattice(noisyMes, coords, or, v1, v2);

%% Helper functions
function plotGrid(coords, or, v1, v2)
    minx = min(coords(1, :));
    maxx = max(coords(1, :));
    miny = min(coords(2, :));
    maxy = max(coords(2, :));
    [x, y] = meshgrid(minx:maxx, miny:maxy);
    xy = [x(:), y(:)];
    T = [v1, v2]';
    xyt = xy * T;
    xt = reshape(xyt(:,1), size(x));
    yt = reshape(xyt(:,2), size(y));
    plot(xt + or(1), yt + or(2), 'r:', 'LineWidth', 0.1)
    hold on;
    plot(xt' + or(1), yt' + or(2), 'r:', 'LineWidth', 0.1)
end

function plotLattice(noisyMes, coords, or, v1, v2)
    % coords is a 2 x n array, n >= 2
    % or is the origin of coordinates
    % v1 and v2 are column vectors, the basis of the lattice
    x = noisyMes(1, :);
    y = noisyMes(2, :);
    scatter(x, y,'linewidth', 4,'MarkerEdgeColor', 'g');
    hold on;
    points = or + coords(1, :) .* v1 + coords(2, :) .* v2;
    x = points(1, :);
    y = points(2, :);
    scatter(x, y, 'x', 'MarkerFaceColor', 'r', 'LineWidth', 2);
    plotGrid(coords, or, v1, v2);
    hold off;
    axis square;
end

function res = cost(x, n, coords, noisyMes)
    res = 0;
    or = [x(1); x(2)];
    v1 = [x(3); x(4)];
    v2 = [x(5); x(6)];
    for z = 1:n
        res = res + norm(or + coords(1, z)*v1 + coords(2, z)*v2 - noisyMes(:, z))^2;
    end
end
