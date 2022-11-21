function [coords, or, v1, v2] = denoiseLattice(noisyMes)
    %DENOISELATTICE Computes the minimum Mean Square Error (MSE) lattice given
    %the input noisy measurements.
    %   Input: noisyMes, a 2 x n measurement vector, where n is >= 2.
    %   Output: coordinates, origin, and vectors of the basis of the denoised lattice.

    %% Plot noisy data
    n = size(noisyMes, 2);
    x = noisyMes(1, :);
    y = noisyMes(2, :);
    scatter(x, y, 'linewidth', 4, 'MarkerEdgeColor', [0 1 0]);

    %% Pick a random inital guess and find distances
    pDist = squareform(pdist(noisyMes'));
    g = 23; % randi(n);
    [~, idx] = sort(pDist(:, g)); % indexes of points from nearest to furthest of g
    or = noisyMes(:, g);
    v1 = noisyMes(:, idx(2)) - or; % Vector from g to its closest neighbor
    v2 = [-v1(2); v1(1)]; % Perpendicular of previous vector

    %% Successive optimization
    coords = zeros(2, n);

    for it = 4:n
        disp("Iteration number " + it)
        % Start optimizing the coordinates
        for z = 1:it
            distInit = norm(or + coords(1, idx(z)) * v1 + coords(2, idx(z)) * v2 - noisyMes(:, idx(z)));
            % The true integer coordinates have to be in a +-1 distance from
            % the floating point ones
            lambdas = [0; 0];

            if (norm(noisyMes(:, idx(z)) - or) > 1e-3)
                lambdas = linsolve([v1, v2], noisyMes(:, idx(z)) - or);
            end

            % I impose coordinates must be between -it and it
            intLambdas = round(lambdas);
            intLambdas(intLambdas > it) = it;
            intLambdas(intLambdas < -it) = -it;

            for i = -1:1

                for j = -1:1
                    auxCoords = [i; j] + intLambdas;
                    distAux = norm(or + [v1, v2] * auxCoords - noisyMes(:, idx(z)));

                    if distAux < distInit
                        distInit = distAux;
                        coords(:, idx(z)) = auxCoords;
                    end

                end

            end

        end

        % Then we optimize vectors and origin
        x0 = [or(1), or(2), v1(1), v1(2), v2(1), v2(2)];
        fCost = @(x) cost(x, it, coords(:, idx(1:it)), noisyMes(:, idx(1:it)));
        [x, totalSqErr] = fminsearch(fCost, x0);
        disp("Total square error: " + totalSqErr)
        or = [x(1); x(2)];
        v1 = [x(3); x(4)];
        v2 = [x(5); x(6)];
        % if it == nIterations
        % plotLattice(noisyMes(:, idx(1:it)), coords(:, idx(1:it)), or, v1, v2);
        % pause(0.1);
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
        xt = reshape(xyt(:, 1), size(x));
        yt = reshape(xyt(:, 2), size(y));
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
        scatter(x, y, 'linewidth', 4, 'MarkerEdgeColor', 'g');
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

        for k = 1:n
            res = res + norm(or + coords(1, k) * v1 + coords(2, k) * v2 - noisyMes(:, k))^2;
        end

    end

end
