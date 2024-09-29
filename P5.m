%%PRACTICA 3
%%El programa selecciona representantes de un área definida
clc;
clear all;
close all;
warning off all;

% Lectura de la imagen "mar.jpg" y obtención de las dimensiones de la imagen
a = imread("playa.jpg");
[m, n, ~] = size(a);    % Obtiene las dimensiones de la imagen en 'm' (alto) y 'n' (ancho)

% Solicitar al usuario el número de clases que desea definir
num_clases = input('Ingrese el número de clases que desea definir: ');

% Crear una nueva figura y mostrar la imagen en ella
figure;
imshow(a);

% Inicializar matrices para almacenar los puntos seleccionados para cada clase
coordenadas_x = cell(1, num_clases);  % Almacena las coordenadas x de cada clase
coordenadas_y = cell(1, num_clases);  % Almacena las coordenadas y de cada clase
colores = ['b', 'r', 'g', 'c', 'm', 'y', 'k'];  % Colores predefinidos para las clases

z = cell(1, num_clases);  % Almacena los valores RGB de cada clase
total_clase = zeros(num_clases, 3);  % Almacena el valor promedio de color de cada clase

% Bucle para seleccionar áreas y puntos de cada clase
for i = 1:num_clases
    fprintf('Seleccione el área para la clase %d\n', i);
    
    % Seleccionar el área para la clase
    [rect, dim_rect] = imcrop(a);
    dim_rect = round(dim_rect);
    
    % Generar coordenadas aleatorias para la clase
    coordenadas_x{i} = randi([dim_rect(1), dim_rect(1)+dim_rect(3)], 1, 100);
    coordenadas_y{i} = randi([dim_rect(2), dim_rect(2)+dim_rect(4)], 1, 100);
    
    % Obtener los valores RGB de los puntos seleccionados
    z{i} = impixel(a, coordenadas_x{i}, coordenadas_y{i});
    
    % Calcular el valor promedio (color) de la clase
    total_clase(i, :) = mean(z{i});
end

for i = 1:num_clases
   % Dibujar los puntos en la imagen con un color diferente para cada clase
    hold on;
    plot(coordenadas_x{i}, coordenadas_y{i}, ['o', colores(mod(i-1, length(colores))+1)], 'Markersize', 5, 'MarkerFaceColor', colores(mod(i-1, length(colores))+1));
    
    % Mostrar una leyenda de las clases
    legendInfo{i} = ['Clase ', num2str(i)]; % Crear la información de la leyenda
    legend(legendInfo); % Mostrar la leyenda con los nombres de las clases
end 

% Bucle para que el usuario seleccione un punto desconocido y clasificarlo
usuario = 0;  % Inicializa la variable 'usuario' en 0 para entrar en el bucle
while usuario == 0  % Mientras 'usuario' sea 0, continuará el bucle
    clear desconocido;
    figure(2);
    desconocido = impixel(a);  % El usuario selecciona un punto en la imagen
    
    figure(3);
    grid on;
    hold on;
    
    % Mostrar los puntos de referencia y el punto desconocido en un espacio 3D
    for i = 1:num_clases
        plot3(total_clase(i, 1), total_clase(i, 2), total_clase(i, 3), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', colores(mod(i-1, length(colores))+1));
    end
    
    plot3(desconocido(1), desconocido(2), desconocido(3), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'k');  % Dibuja el punto desconocido en negro
    
    % Calcular distancias entre el punto desconocido y los puntos de referencia
    distancias = zeros(1, num_clases);
    for i = 1:num_clases
        distancias(i) = norm(total_clase(i, :) - desconocido);
    end
    
    % Encontrar la clase más cercana al punto desconocido
    [~, indice] = min(distancias);
    
    % Mostrar la clasificación del punto desconocido
    fprintf('El punto coincide con la clase %d\n', indice);
    
    % Solicitar al usuario si desea verificar otro punto desconocido o salir
    usuario = input('Introduzca 0 si quiere verificar otro pixel o 1 si quiere salir: ');
end

% Mostrar un mensaje de despedida
disp('Fin del programa, gracias por usarlo');
