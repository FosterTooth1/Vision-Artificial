clc % limpia pantalla
clear all %limpia todo
close all %cierra todo
warning off all

% clases y medias

c1=[0 1 0 3; 0 2 3 0];
c2=[2 3 2 3; 6 6 5 5];
c3=[6 7 7 8; 0 1 3 2];

m1 =mean(c1,2);
m2 =mean(c2,2);
m3 =mean(c3,2);

vec = [3 ; -4];

% matriz de cov utilizando cov() para calcular correctamente
matriz_cov1 = cov(c1');
matriz_cov2 = cov(c2');
matriz_cov3 = cov(c3');

% Inversión de las matrices de covarianza
dato2_1 = inv(matriz_cov1);
dato2_2 = inv(matriz_cov2);
dato2_3 = inv(matriz_cov3);

% Restando los vectores
rest1 = vec - m1;
rest2 = vec - m2;
rest3 = vec - m3;

% Calculando las distancias de Mahalanobis
dato3_1 = sqrt((rest1)' * dato2_1 * (rest1));
dato3_2 = sqrt((rest2)' * dato2_2 * (rest2));
dato3_3 = sqrt((rest3)' * dato2_3 * (rest3));

% Almacenando las distancias y seleccionando la mínima
distancia_mahalanobis = [dato3_1 dato3_2 dato3_3];
disp(distancia_mahalanobis);
[minimo, clase] = min(distancia_mahalanobis);
fprintf("\nMahalanobis \nEl vector pertenece a la clase [%d] \n",clase);

