clc % limpia pantalla
clear all % limpia todo
close all % cierra todo
warning off all

% Preguntar el número de clases y el número de representantes de cada clase
num_clases = input('\n Ingresa el número de clases: ');
num_representantes = input('\n Ingresa el número de representantes por cada clase: ');

% Definimos la paleta de colores
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

num_colores = size(colores, 1); % Número total de colores

% Recursos
centroides = zeros(num_clases,2);
dispersion = zeros(num_clases,2);
Clases = zeros(num_clases, num_representantes,2);

% Preguntar la ubicación de todos los centroides y su dispersión
for i=1:num_clases
    fprintf('\n --- [%d] Centroide --- ',i);
    centroides(i,1) = input('\n Ingresa x: ');
    centroides(i,2) = input('\n Ingresa y: ');
    fprintf('\n --- [%d] Dispersión ---',i);
    dispersion(i,1) = input('\n Ingresa x: ');
    dispersion(i,2) = input('\n Ingresa y: ');
end

% Calculando las distancias entre los centroides
dist_centroides = [];
for i = 1:num_clases
    for j = i+1:num_clases
        dist_centroides = [dist_centroides, norm(centroides(i,:) - centroides(j,:))];
    end
end

% Umbral para la distancia máxima (mitad del promedio de las distancias entre los centroides)
distancia_maxima = (1/2) * mean(dist_centroides);

% Generando las clases
for i=1:num_clases
    Clases(i,:,1) = (randn(1,num_representantes) * dispersion(i,1)) + centroides(i,1);
    Clases(i,:,2) = (randn(1,num_representantes) * dispersion(i,2)) + centroides(i,2);
end

% Grafica de las clases
figure
hold on
grid on

legend_entries = cell(1, num_clases*2 + 1); % Reserva de espacio para las leyendas de las clases, centroides y del vector

for i=1:num_clases
    color_idx = mod(i-1, num_colores) + 1; % Reutiliza los colores cíclicamente
    plot(Clases(i,:,1),Clases(i,:,2),'ko','MarkerFaceColor',colores(color_idx,:),'MarkerSize',10);
    legend_entries{i} = ['Clase ' num2str(i)]; % Guardar la leyenda para cada clase
end

for i=1:num_clases
    % Grafica del centroide
    color_idx = mod(i-1, num_colores) + 1; % Reutiliza los colores cíclicamente
    plot(centroides(i,1),centroides(i,2),'ks','MarkerFaceColor',colores(color_idx,:),'MarkerSize',12,'LineWidth',2);
    legend_entries{num_clases + i} = ['Centroide Clase ' num2str(i)]; % Guardar la leyenda para cada centroide
end

resp_vect = 1;
hVector = [];

while resp_vect == 1

    % Elimina el vector anterior del gráfico si existe
    if ~isempty(hVector)
        delete(hVector);
    end

    % Pide el vector al usuario
    disp('Ingresa los valores del vector :');
    vx = input('x = ');
    vy = input('y = ');
    vector = [vx, vy];

    % Graficando el vector
    hVector = plot(vector(1,1),vector(1,2),'*k','MarkerSize',10);
    legend_entries{num_clases*2 + 1} = 'Vector ingresado'; % Agregar la leyenda para el vector ingresado
    legend(legend_entries); % Crear la leyenda
    grid on

    % Menú para seleccionar el tipo de distancia
    resp_distancia = 1;
    while resp_distancia == 1
        disp('Selecciona el método de distancia:');
        disp('1. Distancia Euclidiana');
        disp('2. Distancia Mahalanobis');
        disp('3. Distancia de Máxima Probabilidad');
        disp('0. Salir');
        disp('En caso de que la distancia mahalobis supere:')
        disp(distancia_maxima)
        disp('El vector no pertenece a ninguna clase')
        opcion = input('Opción: ');

        if opcion == 1
            % Distancia Euclidiana
            distancias = zeros(num_clases, 1);
            for i = 1:num_clases
                distancias(i) = norm(vector - centroides(i,:));  % Distancia al i-ésimo centroide
            end

            [minimo, clase] = min(distancias);
            if minimo <= distancia_maxima
                fprintf('\nEuclidiana \nEl vector pertenece a la clase %d', clase(1));
                fprintf('\nLa mínima distancia es de: %f\n', minimo(1));
            else
                disp('El vector desconocido no pertenece a ninguna clase');
                fprintf('\nLa mínima distancia es de: %f\n', minimo(1));
            end
        elseif opcion == 2
            % Distancia Mahalanobis
            distancias_mahalanobis = zeros(1, num_clases);
            for i = 1:num_clases
                matriz_cov = cov(squeeze(Clases(i,:,:)));
                inv_cov = inv(matriz_cov);
                det_cov = det(matriz_cov);
                rest = vector' - centroides(i,:)';
                distancias_mahalanobis(i) = sqrt((rest)' * inv_cov * rest);
            end
            [minimo, clase] = min(distancias_mahalanobis);
            if minimo <= distancia_maxima
                fprintf('\nMahalanobis \nEl vector pertenece a la clase %d', clase(1));
                fprintf('\nLa mínima distancia es de: %f\n', minimo(1));
            else
                disp('El vector desconocido no pertenece a ninguna clase');
                fprintf('\nLa mínima distancia es de: %f\n', minimo(1));
            end
        elseif opcion == 3
            % Distancia de Máxima Probabilidad
            probabilidades = zeros(1, num_clases);
            for i = 1:num_clases
                matriz_cov = cov(squeeze(Clases(i,:,:)));
                inv_cov = inv(matriz_cov);
                det_cov = det(matriz_cov);
                d = 2; % Dimensión de los datos
                rest = vector' - centroides(i,:)';
                probabilidades(i) = (1 / ((2 * pi)^(d / 2) * sqrt(det_cov))) * exp(-0.5 * rest' * inv_cov * rest);
            end

            % Normalización de probabilidades para que sumen 1
            suma_probabilidades = sum(probabilidades);
            probabilidades_normalizadas = probabilidades / suma_probabilidades;

            [max_prob, clase] = max(probabilidades_normalizadas);
            fprintf("\nMáxima Probabilidad \nEl vector pertenece a la clase [%d]", clase);
            fprintf('\nLa máxima probabilidad normalizada es de: %f\n', max_prob(1));
        elseif opcion == 0
            resp_distancia = 0;
            break;
        else
            disp('Opción no válida. Por favor, selecciona de nuevo.');
        end
    end

    % Elimina el vector anterior del gráfico si existe
    if ~isempty(hVector)
        delete(hVector);
    end

    resp_vect = input('\n¿Deseas probar otra vez con otro vector? Si = 1 / No = 0 : ');

end
hold off

close all

disp('ADIOS');
