clc 
clear all 
close all
warning off all

% Definir colores para graficar puntos
colores = [
    1.0, 0.0, 0.0;  % Rojo
    0.0, 1.0, 0.0;  % Verde
    0.0, 0.0, 1.0;  % Azul
    1.0, 1.0, 0.0;  % Amarillo
    0.0, 1.0, 1.0;  % Cyan
    1.0, 0.0, 1.0;  % Magenta
    0.5, 0.5, 0.5;  % Gris
    1.0, 0.5, 0.0;  % Naranja
    0.5, 0.0, 0.5;  % Púrpura
    0.0, 0.5, 0.5;  % Verde azulado
];

% Preguntar el número de clases y el número de representantes de cada clase
num_clases = input('\n Ingresa el número de clases: ');
num_representantes = input('\n Ingresa el número de representantes por cada clase: ');

% Inicializar matrices para guardar los valores RGB de las clases
Clases_RGB = zeros(num_clases, num_representantes, 3); % Almacena los valores RGB de cada clase
PUNTOS_X = zeros(num_clases,num_representantes);
PUNTOS_Y = zeros(num_clases,num_representantes);

legend_entries = cell(1, num_clases + 1); % Reserva de espacio para las leyendas de las clases, centroides y del vector

% Leer y mostrar imagen
% h = imread('peppers.png');
h = imread('mar.jpg');

% Mostrar imagen y seleccionar la región (rectángulo)
figure; imshow(h);
hold on;

% Generar las clases
for i=1:num_clases
    title(sprintf('Selecciona la región de la clase %d', i));  % Actualiza el título correctamente
    [rect, dim_rect] = imcrop(h);
    dim_rect1 = round(dim_rect); % Redondear los valores de la región seleccionada

    % Generar puntos aleatorios dentro de la región seleccionada
    puntos_x = randi([dim_rect1(1), dim_rect1(1) + dim_rect1(3)], 1, num_representantes); % Coordenadas x
    puntos_y = randi([dim_rect1(2), dim_rect1(2) + dim_rect1(4)], 1, num_representantes); % Coordenadas y
    
    % Guardar los puntos x & y
    PUNTOS_X(i,:) = puntos_x;
    PUNTOS_Y(i,:) = puntos_y;

    % Obtener los valores RGB
    Clases_RGB(i,:,:) = impixel(h,puntos_x(:),puntos_y(:));
end

% Graficar los puntos de las clases
for i=1:num_clases
    color_idx = mod(i-1,size(colores,1))+1;
    plot(PUNTOS_X(i,:),PUNTOS_Y(i,:),'o','MarkerSize',10,'MarkerFaceColor',colores(color_idx,:));
    legend_entries{i} = ['Clase ' num2str(i)]; % Guardar la leyenda para cada clase
end

title('Puntos de las clases');

% Calcular los centroides de los Clases_RGB 
Centroides_RGB = zeros(num_clases,3);

for i = 1:num_clases
    Centroides_RGB(i,:) = mean(squeeze(Clases_RGB(i,:,:)),1);
end

% Mostrar los centroides para comprobar
for i=1:num_clases
    fprintf("\n Centroide [%i] = %d %d %d",i,Centroides_RGB(i,:));
end

% Calcular distancias entre centroides para usar un umbral
dist_Centroides_RGB = [];
for i = 1:num_clases
    for j = i+1:num_clases
        dist_Centroides_RGB = [dist_Centroides_RGB, norm(Centroides_RGB(i,:) - Centroides_RGB(j,:))];
    end
end

% Calcular la distancia máxima promedio como umbral
if isempty(dist_Centroides_RGB)
    distancia_maxima = 0; % Si solo hay una clase
else
    distancia_maxima = (1/2) * mean(dist_Centroides_RGB); % Umbral de clasificación
end

% Inicializar variable para el punto a ingresar
hVector = [];
resp_vect = 1;

while resp_vect == 1
    %Seleccionar el punto para clasificarlo
    disp("\n");
    disp('Selecciona el punto en la imagen: ');
    [vx,vy] = ginput(1); % Permite seleccionar un punto en la imagen
    vector_RGB = impixel(h,vx,vy); % Obtener los valores RGB del pixel
    fprintf("\n Ingresaste el punto RGB : %d %d %d",vector_RGB);

    % Eliminar el punto anterior si existe
    if ~isempty(hVector)
        delete(hVector);
    end

    % Graficar el vector seleccionado
    hVector = plot(vx,vy,'*k','MarkerSize',10);
    legend_entries{num_clases + 1} = 'Punto ingresado';
    legend(legend_entries);
    
    %% Calcular las distancias con funciones

    % EUCLIDEANA
    distancias_Euclidena_RGB = zeros(num_clases,1);
    for i = 1:num_clases
        distancias_Euclidena_RGB(i) = norm(vector_RGB - Centroides_RGB(i,:));
    end
    
    % Sacar la distancia mínima con su índice
    [minimo, clase] = min(distancias_Euclidena_RGB);

    if minimo <= distancia_maxima
        fprintf('\n\n -> EUCLIDEANA \nEl vector pertenece a la CLASE[%d]', clase);
        fprintf('\nLa mínima distancia es de: %f\n', minimo);
    else
        fprintf('\nEl vector desconocido no pertenece a ninguna clase');
        fprintf('\nLa mínima distancia es de: %f\n', minimo);
    end
    
    % MAHALANOBIS
    distancias_Mahalanobis_RGB = zeros(1,num_clases);
    for i=1:num_clases
        matriz_cov = cov(squeeze(Clases_RGB(i,:,:)));
        inv_cov = inv(matriz_cov);
        rest = vector_RGB' - Centroides_RGB(i,:)';
        distancias_Mahalanobis_RGB(i) = sqrt((rest)' * inv_cov* rest);
    end
    [minimo_2,clase_2] = min(distancias_Mahalanobis_RGB);
    if minimo_2 <= distancia_maxima
        fprintf('\n\n -> MAHALANOBIS \nEl vector pertenece a la CLASE[%d]', clase_2);
        fprintf('\nLa mínima distancia es de: %f\n', minimo_2);
    else
        fprintf('\nEl vector desconocido no pertenece a ninguna clase');
        fprintf('\nLa mínima distancia es de: %f\n', minimo_2);
    end

    % Máxima Probabilidad
    probabilidades = zeros(1,num_clases);
    for i=1:num_clases
        matriz_cov = cov(squeeze(Clases_RGB(i,:,:)));
        inv_cov = inv(matriz_cov);
        det_cov = det(matriz_cov);
        d = 3; % Dimensión de los datos RGB
        rest = vector_RGB' - Centroides_RGB(i,:)';
        probabilidades(i) = (1 / ((2*pi)^(d/2)*sqrt(det_cov))) * exp(-0.5 * rest' * inv_cov * rest);
    end

    sum_probabilidades = sum(probabilidades);
    probabilidades_normalizadas = probabilidades / sum_probabilidades;
    [max_prob,clase_3] = max(probabilidades_normalizadas);
    fprintf("\n\n -> MÁXIMA PROBABILIDAD \nEl vector pertenece a la CLASE[%d]", clase_3);
    fprintf('\nLa máxima probabilidad normalizada es de: %f\n', max_prob);

    
    % Pregunta si se quiere ingresar otro vector
    resp_vect = input('\nDeseas ingresar otro vector? (1: Sí, 0: No): ');

    % Opción no válida
    if resp_vect ~= 1 && resp_vect ~=0
        fprintf('\nOpción no válida');
    end
end

hold off;
close all;
disp('ADIOS');
