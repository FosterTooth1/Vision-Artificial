clc;
clear;
close all;

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

% Convertir la imagen a escala de grises para segmentar las letras
grayImg = rgb2gray(img);

% Umbralización para separar las letras del fondo
binaryMask = grayImg < 200; % Ajustar el umbral según la intensidad del fondo

% Aplicar la máscara a la imagen
maskedImg = img;
maskedImg(repmat(~binaryMask, [1 1 3])) = 255; % Convertir el fondo a blanco

% Mostrar la imagen segmentada
figure;
imshow(maskedImg);
title('Imagen con letras segmentadas');

% Convertir la imagen segmentada a un vector de píxeles (2D)
[m, n, p] = size(maskedImg);
pixels = double(reshape(maskedImg, m * n, p));

% Filtrar píxeles blancos (fondo)
nonWhitePixels = pixels(~all(pixels == 255, 2), :);

% Solicitar el número de representantes
numRepresentatives = input('Ingrese la cantidad de representantes a plotear: ');
while numRepresentatives <= 0 || floor(numRepresentatives) ~= numRepresentatives || numRepresentatives > size(nonWhitePixels, 1)
    disp('El número de representantes debe ser un entero positivo menor o igual al total de píxeles no blancos.');
    numRepresentatives = input('Ingrese la cantidad de representantes a plotear: ');
end

% Solicitar el número de clusters
numClusters = input('Ingrese el número de clusters (clases): ');
while numClusters <= 0 || floor(numClusters) ~= numClusters
    disp('El número de clusters debe ser un entero positivo.');
    numClusters = input('Ingrese el número de clusters (clases): ');
end

% Inicialización aleatoria de representantes
rng('default'); % Para reproducibilidad
indices = randperm(size(nonWhitePixels, 1), numRepresentatives);
selectedPixels = nonWhitePixels(indices, :);

% Inicialización aleatoria de centroides
centroids = selectedPixels(randperm(size(selectedPixels, 1), numClusters), :);

% Coordenadas originales de los representantes
[yCoords, xCoords] = ind2sub([m, n], find(~all(maskedImg == 255, 3)));
selectedCoords = [yCoords(indices), xCoords(indices)];

% Variables para la iteración
maxIterations = 100; % Definir un límite máximo
prevCentroids = zeros(size(centroids));
idx = zeros(size(selectedPixels, 1), 1); % Etiquetas de cada punto representante
iteration = 0;

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
    colors = hsv(numClusters); % Generar colores para los clústeres
    for k = 1:numClusters
        scatter(selectedCoords(idx == k, 2), selectedCoords(idx == k, 1), 30, colors(k, :), 'filled');
    end
    hold off;
    title(['Iteración ', num2str(iteration), ': Agrupación de puntos en la imagen']);
    drawnow;

    % Mostrar los centroides en un cubo RGB
    figure;
    scatter3(centroids(:, 1), centroids(:, 2), centroids(:, 3), 100, centroids / 255, 'filled');
    xlabel('Canal R');
    ylabel('Canal G');
    zlabel('Canal B');
    grid on;
    xlim([0 255]);
    ylim([0 255]);
    zlim([0 255]);
    title(['Centroides en espacio RGB - Iteración ', num2str(iteration)]);
end

% Mostrar la cantidad de iteraciones realizadas
disp(['K-means completado en ', num2str(iteration), ' iteraciones.']);

