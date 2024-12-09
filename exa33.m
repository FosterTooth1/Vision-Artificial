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
title('Imagen Original');

% Convertir la imagen a un vector de píxeles
[m, n, p] = size(img); % Tamaño de la imagen
pixels = double(reshape(img, m * n, p)); % Convertir a Nx3 (RGB)

% Filtrar píxeles blancos (considerados como fondo)
threshold = 250; % Umbral para considerar un píxel como blanco
disp(['Umbral utilizado para eliminar el fondo blanco: ', num2str(threshold)]);
mask = ~(pixels(:, 1) > threshold & pixels(:, 2) > threshold & pixels(:, 3) > threshold);
filteredPixels = pixels(mask, :); % Mantener solo píxeles que no sean blancos

% Coordenadas originales de los píxeles filtrados
originalIndices = find(mask); % Índices de los píxeles que no son blancos
[yCoords, xCoords] = ind2sub([m, n], originalIndices);

% Solicitar el número de representantes
numRepresentatives = input('Ingrese la cantidad de puntos representantes para la imagen: ');
while numRepresentatives <= 0 || floor(numRepresentatives) ~= numRepresentatives || numRepresentatives > size(filteredPixels, 1)
    disp('El número de representantes debe ser un entero positivo menor o igual al total de píxeles no blancos.');
    numRepresentatives = input('Ingrese la cantidad de puntos representantes para la imagen: ');
end

% Seleccionar puntos representantes aleatoriamente de los píxeles no blancos
rng('default'); % Para reproducibilidad
indices = randperm(size(filteredPixels, 1), numRepresentatives);
selectedPixels = filteredPixels(indices, :);

% Inicialización del primer centroide con el primer punto
centroids = selectedPixels(1, :);
numClusters = 1;

% Etiquetas para los puntos representantes
idx = zeros(size(selectedPixels, 1), 1);

% Colores para las clases
classColors = generateColors(numClusters);

% Iterar sobre los puntos y asignar a clústeres
distanceThreshold = 100; % Umbral para crear nuevos clústeres
disp(['Umbral para crear nuevos clústeres: ', num2str(distanceThreshold)]);
disp('Ejecutando K-means con umbral establecido...');
for i = 1:size(selectedPixels, 1)
    distances = sqrt(sum((centroids - selectedPixels(i, :)).^2, 2)); % Distancia euclidiana a todos los centroides
    [minDist, closestCentroid] = min(distances);
    
    if minDist > distanceThreshold
        % Crear un nuevo clúster si la distancia es mayor que el umbral
        numClusters = numClusters + 1;
        centroids = [centroids; selectedPixels(i, :)]; % Agregar un nuevo centroide
        idx(i) = numClusters;
        classColors = generateColors(numClusters); % Actualizar colores para las clases
    else
        % Asignar al clúster más cercano
        idx(i) = closestCentroid;
    end
end

% Contar el número de representantes en cada clase
classCounts = zeros(numClusters, 1);
for k = 1:numClusters
    classCounts(k) = sum(idx == k);
end

% Identificar las dos clases con más representantes
[~, sortedIndices] = sort(classCounts, 'descend');
topClasses = sortedIndices(1:2);

% Filtrar para las dos clases principales
filteredIdx = idx;
filteredIdx(~ismember(idx, topClasses)) = 0; % Asignar a clase 0 si no está en las dos principales

% Recalcular centroides finales para las dos clases principales
filteredCentroids = zeros(2, size(centroids, 2));
for k = 1:2
    clusterPoints = selectedPixels(filteredIdx == topClasses(k), :);
    if ~isempty(clusterPoints)
        filteredCentroids(k, :) = mean(clusterPoints, 1); % Recalcular centroide
    end
end

% Sumar representantes de las clases no consideradas a la clase con menos representantes
excludedClasses = setdiff(1:numClusters, topClasses);
excludedRepresentatives = [];
for k = excludedClasses
    excludedRepresentatives = [excludedRepresentatives; selectedPixels(idx == k, :)];
end

if classCounts(topClasses(1)) > classCounts(topClasses(2))
    % Sumar representantes excluidos a la clase 2
    filteredIdx(ismember(idx, excludedClasses)) = topClasses(2);
    filteredCentroids(2, :) = mean([filteredCentroids(2, :); excludedRepresentatives], 1);
else
    % Sumar representantes excluidos a la clase 1
    filteredIdx(ismember(idx, excludedClasses)) = topClasses(1);
    filteredCentroids(1, :) = mean([filteredCentroids(1, :); excludedRepresentatives], 1);
end

% Mostrar agrupación de colores en la imagen (solo dos clases principales)
figure;
imshow(img);
hold on;
legends = cell(1, 2);
topClassColors = generateColors(2);
for k = 1:2
    scatter(xCoords(indices(filteredIdx == topClasses(k))), ...
            yCoords(indices(filteredIdx == topClasses(k))), ...
            15, topClassColors(k, :), 'filled', 'MarkerEdgeColor', 'k');
    legends{k} = sprintf('Clase %d (Representantes: %d)', topClasses(k), ...
                         sum(filteredIdx == topClasses(k)));
end
hold off;
title('Agrupación de las 2 clases principales en la imagen sin fondo blanco');

% Añadir la leyenda
legend(legends, 'Location', 'bestoutside');

disp('Programa finalizado. Mostrando solo las 2 clases principales.');

