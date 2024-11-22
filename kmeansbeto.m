% K-means clustering implementado desde cero con selección de representantes
clc;
clear;
close all;

% Función para generar colores únicos para cada clase
generateColors = @(numClusters) hsv(numClusters);

% Leer la imagen
[file, path] = uigetfile({'*.jpg;*.png;*.bmp', 'Archivos de imagen (*.jpg, *.png, *.bmp)'}, ...
                         'Seleccione una imagen para procesar');
if isequal(file, 0)
    disp('No se seleccionó ninguna imagen. Saliendo...');
    return;
end
img = imread(fullfile(path, file));

% Mostrar la imagen original
figure;
imshow(img);
title('Imagen original');

% Convertir la imagen a un vector de píxeles (2D)
[m, n, p] = size(img); % Tamaño de la imagen
pixels = double(reshape(img, m * n, p)); % Convertir a Nx3 (RGB)

% Solicitar el número de clusters
numClusters = input('Ingrese el número de clusters (clases): ');
while numClusters <= 0 || floor(numClusters) ~= numClusters
    disp('El número de clusters debe ser un entero positivo.');
    numClusters = input('Ingrese el número de clusters (clases): ');
end

% Solicitar el número de representantes
numRepresentatives = input('Ingrese la cantidad de puntos representantes para la imagen: ');
while numRepresentatives <= 0 || floor(numRepresentatives) ~= numRepresentatives || numRepresentatives > size(pixels, 1)
    disp('El número de representantes debe ser un entero positivo menor o igual al total de píxeles.');
    numRepresentatives = input('Ingrese la cantidad de puntos representantes para la imagen: ');
end

% Solicitar el número máximo de iteraciones
maxIterations = input('Ingrese el número máximo de iteraciones: ');
while maxIterations <= 0 || floor(maxIterations) ~= maxIterations
    disp('El número máximo de iteraciones debe ser un entero positivo.');
    maxIterations = input('Ingrese el número máximo de iteraciones: ');
end

% Seleccionar puntos representantes aleatoriamente
rng('default'); % Para reproducibilidad
indices = randperm(size(pixels, 1), numRepresentatives);
selectedPixels = pixels(indices, :);

% Inicialización aleatoria de centroides
centroids = selectedPixels(randperm(size(selectedPixels, 1), numClusters), :);

% Variables para la iteración
prevCentroids = zeros(size(centroids));
idx = zeros(size(selectedPixels, 1), 1); % Etiquetas de cada punto representante
iteration = 0;

% Colores para las clases
classColors = generateColors(numClusters);

% Coordenadas originales de los representantes
[yCoords, xCoords] = ind2sub([m, n], indices);

% Ejecutar K-means
while ~isequal(centroids, prevCentroids) && iteration < maxIterations
    iteration = iteration + 1;
    prevCentroids = centroids;
    
    % Asignar cada representante al clúster más cercano
    for i = 1:size(selectedPixels, 1)
        distances = sum((centroids - selectedPixels(i, :)).^2, 2); % Distancia euclidiana
        [~, idx(i)] = min(distances); % Índice del centroide más cercano
    end
    
    % Actualizar centroides
    for k = 1:numClusters
        clusterPoints = selectedPixels(idx == k, :); % Representantes asignados al clúster k
        if ~isempty(clusterPoints)
            centroids(k, :) = mean(clusterPoints, 1); % Nuevo centroide
        end
    end
    
    % Mostrar agrupación de colores en la iteración actual
    figure;
    imshow(img);
    hold on;
    for k = 1:numClusters
        scatter(xCoords(idx == k), yCoords(idx == k), 15, classColors(k, :), 'filled', 'MarkerEdgeColor', 'k');
    end
    hold off;
    title(['Iteración ', num2str(iteration), ': Agrupación de puntos en la imagen']);
    drawnow;
end

disp(['K-means completado en ', num2str(iteration), ' iteraciones.']);