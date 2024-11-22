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

% Asignar un umbral fijo
threshold = 100; % Umbral fijo para la distancia entre puntos y centroides
%disp(['Umbral fijo establecido en: ', num2str(threshold)]);

% Seleccionar puntos representantes aleatoriamente
rng('default'); % Para reproducibilidad
numRepresentatives = input('Ingrese la cantidad de puntos representantes para la imagen: ');
while numRepresentatives <= 0 || floor(numRepresentatives) ~= numRepresentatives || numRepresentatives > size(pixels, 1)
    disp('El número de representantes debe ser un entero positivo menor o igual al total de píxeles.');
    numRepresentatives = input('Ingrese la cantidad de puntos representantes para la imagen: ');
end
indices = randperm(size(pixels, 1), numRepresentatives);
selectedPixels = pixels(indices, :);

% Inicialización del primer centroide con el primer punto
centroids = selectedPixels(1, :);
numClusters = 1;

% Etiquetas para los puntos representantes
idx = zeros(size(selectedPixels, 1), 1);

% Coordenadas originales de los representantes
[yCoords, xCoords] = ind2sub([m, n], indices);

% Colores para las clases
classColors = generateColors(numClusters);

% Iterar sobre los puntos y asignar a clústeres
disp('Ejecutando K-means con umbral fijo...');
for i = 1:size(selectedPixels, 1)
    distances = sqrt(sum((centroids - selectedPixels(i, :)).^2, 2)); % Distancia euclidiana a todos los centroides
    [minDist, closestCentroid] = min(distances);
    
    if minDist > threshold
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

% Recalcular centroides finales
for k = 1:numClusters
    clusterPoints = selectedPixels(idx == k, :);
    if ~isempty(clusterPoints)
        centroids(k, :) = mean(clusterPoints, 1); % Recalcular centroide
    end
end

% Mostrar resultados
disp(['Número de clases detectadas: ', num2str(numClusters)]);

% Mostrar agrupación de colores en la imagen
figure;
imshow(img);
hold on;
legends = cell(1, numClusters);
for k = 1:numClusters
    scatter(xCoords(idx == k), yCoords(idx == k), 15, classColors(k, :), 'filled', 'MarkerEdgeColor', 'k');
    numPointsInClass = sum(idx == k); % Contar puntos en cada clase
    legends{k} = sprintf('Clase %d (%d puntos)', k, numPointsInClass); % Crear leyenda con número de puntos
end
hold off;
title(['Agrupación de puntos en la imagen con ', num2str(numClusters), ...
       ' clases (', num2str(numRepresentatives), ' representantes)']);

% Añadir la leyenda
legend(legends, 'Location', 'bestoutside');

% Reconstruir la imagen segmentada
finalIdx = zeros(size(pixels, 1), 1);
for i = 1:size(pixels, 1)
    distances = sum((centroids - pixels(i, :)).^2, 2);
    [~, finalIdx(i)] = min(distances);
end
segmentedImg = reshape(centroids(finalIdx, :), [m, n, p]);

% Convertir a uint8 para visualizar
segmentedImg = uint8(segmentedImg);

disp('Programa finalizado.');
