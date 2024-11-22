clc;
close all;
clear all;
warning off all;

% Cargar la imagen
img = imread('peppers.png');
[X, Y, Z] = size(img);

% Solicitar número de clases y representantes por clase
nClases = input('Ingrese el número de clases (n): ');
mRepresentantes = input('Ingrese el número de representantes por clase (m): ');

% Convertir la imagen en un arreglo de puntos RGB
datosRGB = double(reshape(img, [], Z)); % Convertir la imagen a una lista de valores RGB

% K-means para clasificar los colores en n clases
[idx, centroides] = kmeans(datosRGB, nClases, 'MaxIter', 200);

% Crear conjuntos de representantes equilibrados
representantes = [];
for i = 1:nClases
    % Filtrar los puntos pertenecientes a la clase actual
    puntosClase = datosRGB(idx == i, :);
    
    % Seleccionar m puntos de forma aleatoria
    if size(puntosClase, 1) >= mRepresentantes
        indices = randperm(size(puntosClase, 1), mRepresentantes);
    else
        indices = 1:size(puntosClase, 1); % Si hay menos de m puntos, usarlos todos
    end
    representantes = [representantes; puntosClase(indices, :)];
end

% Asignar etiquetas para cada clase
etiquetas = repelem(1:nClases, mRepresentantes)';

% Mostrar los centroides
disp('Centroides de las clases:');
disp(centroides);

% Crear imagen coloreada según las clases
imagenClases = reshape(idx, X, Y); % Convertir los índices de clases en una matriz 2D
colores = lines(nClases); % Generar colores para las clases
imgColoreada = zeros(X, Y, 3);

for i = 1:nClases
    % Asignar el color correspondiente a los píxeles de cada clase
    for c = 1:3
        imgColoreada(:, :, c) = imgColoreada(:, :, c) + ...
            (imagenClases == i) * colores(i, c);
    end
end

% Mostrar la imagen original y la segmentada por clases
figure;
subplot(1, 2, 1);
imshow(img);
title('Imagen Original');

subplot(1, 2, 2);
imshow(imgColoreada);
title('Imagen Segmentada por Clases');

% Interacción para seleccionar puntos
continuar = true;
while continuar
    figure, imshow(img);
    title('Haga clic en un punto para clasificarlo o cierre la ventana para salir.');
    
    [x, y] = ginput(1); % Obtener coordenadas del punto seleccionado
    
    if isempty(x) || isempty(y)
        disp('No se seleccionó ningún punto. Finalizando...');
        break;
    end
    
    x = round(x);
    y = round(y);
    
    if x > 0 && x <= Y && y > 0 && y <= X
        % Obtener valores RGB del píxel seleccionado
        pixelSeleccionado = double(squeeze(img(y, x, :)))';
        
        % Clasificar el punto utilizando la distancia al centroide
        distancias = sqrt(sum((centroides - pixelSeleccionado).^2, 2));
        [~, clase] = min(distancias);
        
        % Mostrar la clase del punto
        fprintf('El punto seleccionado pertenece a la Clase %d.\n', clase);
    else
        disp('Coordenadas fuera de la imagen.');
    end
    
    % Preguntar al usuario si desea continuar
    respuesta = input('¿Desea clasificar otro punto? (s/n): ', 's');
    if lower(respuesta) ~= 's'
        continuar = false;
    end
end

disp('Programa finalizado.');