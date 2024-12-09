clc;
clear;
close all;
warning off all;

% Leer la imagen
[file, path] = uigetfile({'*.jpg;*.png;*.bmp', 'Archivos de imagen (*.jpg, *.png, *.bmp)'}, ...
                         'Seleccione una imagen para procesar');
if isequal(file, 0)
    disp('No se seleccionó ninguna imagen. Saliendo...');
    return;
end
img = imread(fullfile(path, file));

% Mostrar la imagen
figure(1);
imshow(img);
title(['Imagen seleccionada: ', file]);

% Solicitar el número de clases y puntos representativos
num_clases = 2;
num_representantes = input('Introduce el número de representantes por clase: ');
lr = input('Introduce el valor de r (tasa de aprendizaje): ');
max_iter = input('Introduce el número máximo de iteraciones: ');

% Solicitar los pesos iniciales
disp('Introduce los pesos iniciales (formato [w1, w2, w3, w4]):');
w = input('Pesos iniciales: ')';

% Inicializar variables
representantes_rgb = cell(1, 2);
representantes_coords = cell(1, 2);
X = []; % Matriz de características
Y = []; % Vector de etiquetas
w_prev = w;

% Seleccionar las áreas y generar representantes para cada clase
colores = lines(num_clases);
figure(2);
imshow(img);
hold on;

for i = 1:num_clases
    disp(['Selecciona el área para la clase ', num2str(i), ':']);
    rect = round(getrect); 
    region = imcrop(img, rect);
    [r, c, ~] = size(region);
    representantes_x = randi([1, c], 1, num_representantes);
    representantes_y = randi([1, r], 1, num_representantes);
    representantes_rgb{i} = impixel(region, representantes_x, representantes_y);
    X = [X; [representantes_rgb{i}, ones(num_representantes, 1)]];
    
    % Asignar etiquetas alternando entre 1 y -1
    Y = [Y; (-1)^(i + 1) * ones(num_representantes, 1)];

    representantes_coords{i} = [representantes_x + rect(1); representantes_y + rect(2)];
    plot(representantes_coords{i}(1,:), representantes_coords{i}(2,:), 'o', 'MarkerSize', 10, 'MarkerFaceColor', colores(i,:))
    % Crear las etiquetas para la leyenda
    legendStrings = arrayfun(@(i) sprintf('Clase %d', i), 1:2, 'UniformOutput', false);
    legend([legendStrings]);
end

% Clasificador basado en pesos y condición de parada por iteraciones
for iter = 1:max_iter
    for i = 1:size(X, 1)
        x_i = X(i, :)'; 
        y_i = Y(i);
        output = w' * x_i;
        
        % Actualización de pesos si la clasificación es incorrecta
        if y_i * output <= 0
            w = w + lr * y_i * x_i;
        end
    end

    % Criterio de parada basado en el número de iteraciones
    if iter >= max_iter
        disp('Número máximo de iteraciones alcanzado.');
        break;
    end
end

disp('Pesos finales:');
disp(w);

% Clasificación de puntos según clases
repeat = true;
while repeat
    disp('Selecciona un píxel para clasificar:');
    [x, y] = ginput(1);
    P = impixel(img, round(x), round(y));
    P = [P, 1]; % Agregar sesgo

    % Determinar la clase
    result_P = w' * P';
    if result_P < 0
        clase_asignada = 'Clase 2';
    else
        clase_asignada = 'Clase 1';
    end
    fprintf('El píxel seleccionado pertenece a %s\n', clase_asignada);

    % Mostrar el punto seleccionado en la imagen
    color_aleatorio = randi([0, 255], 1, 3) / 255;
    plot(x, y, 'o', 'MarkerSize', 10, 'MarkerFaceColor', color_aleatorio, 'MarkerEdgeColor', 'k');

    % Gráfica 3D de los puntos RGB y el plano de separación
    figure(3);
    hold on;
    grid on;

    % Plotear los puntos de cada clase en la gráfica 3D
    plot3(representantes_rgb{1}(:, 1), representantes_rgb{1}(:, 2), representantes_rgb{1}(:, 3), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'red');
    plot3(representantes_rgb{2}(:, 1), representantes_rgb{2}(:, 2), representantes_rgb{2}(:, 3), 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'blue');

    % Crear el plano de separación
    [x_grid, y_grid] = meshgrid(0:10:255, 0:10:255); % Ajuste de malla en el rango RGB (0 a 255)
    z_grid = -(w(1) * x_grid + w(2) * y_grid + w(4)) / w(3);  % Calcular z_grid como matriz
    
    surf(x_grid, y_grid, z_grid, 'FaceAlpha', 0.5, 'EdgeColor', 'none'); % Graficar el plano de separación

    xlabel('R');
    ylabel('G');
    zlabel('B');
    title('Clases en 3D con plano de separación');
    legend({'Clase 1 (Rojo)', 'Clase 2 (Azul)'}, 'Location', 'best');

    view(3);  
    grid on;
    axis([0 255 0 255 0 255]);
    hold off;

    % Preguntar si se quiere probar con otro píxel
    respuesta = input('¿Quieres probar con otro píxel? (s/n): ', 's');
    if strcmpi(respuesta, 'n')
        repeat = false;
        disp('Fin del proceso...');
    end
end