clc % limpia pantalla
clear all % limpia todo
close all % cierra todo
warning off all

% Clases y medias
c1 = [0 1 0 3; 0 2 3 0];
c2 = [4 4 4 5; 0 3 2 6];
c3 = [6 7 7 8; 0 1 3 2];

m1 = [1; 1.25]; % media de la clase 1
m2 = [4.25; 2.75]; % media de la clase 2
m3 = [7; 1.5]; % media de la clase 3

% Punto que queremos clasificar
vec = [3; -4];

% Calculamos la distancia euclidiana para cada clase
distancia_euclidiana_1 = sqrt(sum((vec - m1).^2)); % distancia a la clase 1
distancia_euclidiana_2 = sqrt(sum((vec - m2).^2)); % distancia a la clase 2
distancia_euclidiana_3 = sqrt(sum((vec - m3).^2)); % distancia a la clase 3

% Almacenamos las distancias
distancia_euclidian = [distancia_euclidiana_1 distancia_euclidiana_2 distancia_euclidiana_3];

% Mostramos las distancias
disp(distancia_euclidian);

% Encontramos la clase con la menor distancia
[minimo, clase] = min(distancia_euclidian);

% Mostramos la clase a la que pertenece el vector
fprintf("\nEuclidiana \nEl vector pertenece a la clase [%d] \n", clase);
