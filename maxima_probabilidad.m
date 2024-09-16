clc % limpia pantalla
clear all % limpia todo
close all % cierra todo
warning off all

% Clases y medias
c1=[0 1 0 3; 0 2 3 0];
c2=[2 3 2 3; 6 6 5 5];
c3=[6 7 7 8; 0 1 3 2];

m1 =mean(c1,2);
m2 =mean(c2,2);
m3 =mean(c3,2);

% Punto que queremos clasificar
vec = [3; -4];

% Matrices de covarianza de cada clase
cov1 = cov(c1');
cov2 = cov(c2');
cov3 = cov(c3');

% Inversas de las matrices de covarianza
inv_cov1 = inv(cov1);
inv_cov2 = inv(cov2);
inv_cov3 = inv(cov3);

% Determinantes de las matrices de covarianza
det_cov1 = det(cov1);
det_cov2 = det(cov2);
det_cov3 = det(cov3);

% Dimensión de los datos (2 dimensiones: x1, x2)
d = 2;

% Calculamos la función de probabilidad para cada clase
% P(x | mu, Sigma) = (1 / ((2*pi)^(d/2) * sqrt(|Sigma|))) * exp(-0.5 * (x - mu)' * inv(Sigma) * (x - mu))

prob1 = (1 / ((2 * pi)^(d / 2) * sqrt(det_cov1))) * exp(-0.5 * (vec - m1)' * inv_cov1 * (vec - m1));
prob2 = (1 / ((2 * pi)^(d / 2) * sqrt(det_cov2))) * exp(-0.5 * (vec - m2)' * inv_cov2 * (vec - m2));
prob3 = (1 / ((2 * pi)^(d / 2) * sqrt(det_cov3))) * exp(-0.5 * (vec - m3)' * inv_cov3 * (vec - m3));

% Almacenamos las probabilidades
probabilidades = [prob1, prob2, prob3];

% Mostramos las probabilidades
disp(probabilidades);

% Encontramos la clase con la mayor probabilidad
[max_prob, clase] = max(probabilidades);

% Mostramos la clase a la que pertenece el vector
fprintf("\nMáxima Probabilidad \nEl vector pertenece a la clase [%d] \n", clase);
