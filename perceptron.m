clc
close all
clear all
warning off all

% Datos de entrada en 3D para un cubo (8 vértices)
%X = [0 0 0; 0 0 1; 0 1 0; 0 1 1; 1 0 0; 1 0 1; 1 1 0; 1 1 1];
%y = [0; 1; 1; 1; 0; 0; 0; 1];

X = [0 0 0; 1 0 1; 1 0 0; 1 1 0; 0 0 1; 0 1 1; 0 1 0; 1 1 1];
y = [0; 0; 0; 0; 1; 1; 1; 1];

% Solicitar pesos, bias y tasa de aprendizaje iniciales al usuario
pesos = input('Ingrese los pesos iniciales como un vector columna [w1; w2; w3]: ');
bias = input('Ingrese el valor inicial del bias: ');
learning_rate = input('Ingrese la tasa de aprendizaje inicial: ');
max_iteraciones = 1000;

itera = 0;
converged = false;

% Entrenamiento del perceptrón
while ~converged && itera < max_iteraciones
    itera = itera + 1;
    converged = true;
    
    for i = 1:size(X, 1)
        % Calcular la salida del perceptrón
        output = pesos' * X(i,:)' + bias;
        
        % Verificar si la clasificación es incorrecta y actualizar pesos y sesgo
        if y(i) == 1 && output <= 0  % Caso: pertenece a clase C2 pero está mal clasificado
            converged = false;
            pesos = pesos + learning_rate * X(i,:)';
            bias = bias + learning_rate;
        elseif y(i) == 0 && output >= 0  % Caso: pertenece a clase C1 pero está mal clasificado
            converged = false;
            pesos = pesos - learning_rate * X(i,:)';
            bias = bias - learning_rate;
        end
    end
end

% Mostrar el número de iteraciones y los pesos ajustados
disp(['No.final iteraciones: ', num2str(itera)]);
disp('Pesos:');
disp(pesos);
disp(['Bias final: ', num2str(bias)]);

% Mostrar la función de solución
fprintf('Funcion de solucion = %.4fX + %.4fY + %.4fZ + %.4fD\n', pesos(1), pesos(2), pesos(3), bias);


% Graficar los puntos, el cubo, y el plano de separación
figure;
hold on;

% Graficar el cubo en 3D (aristas)
cubo_aristas = [
    1 1 1; 1 1 0;
    1 1 1; 1 0 1;
    1 1 1; 0 1 1;
    0 0 0; 0 0 1;
    0 0 0; 0 1 0;
    0 0 0; 1 0 0;
    1 0 0; 1 0 1;
    1 0 0; 1 1 0;
    0 1 1; 0 1 0;
    0 1 1; 1 1 1;
    0 0 1; 0 1 1;
    0 0 1; 1 0 1
];

for i = 1:2:size(cubo_aristas, 1)
    plot3(cubo_aristas(i:i+1, 1), cubo_aristas(i:i+1, 2), cubo_aristas(i:i+1, 3), 'k-', 'LineWidth', 1.5);
end

% Graficar los puntos de datos en 3D
for i = 1:size(X, 1)
    if y(i) == 0
        plot3(X(i, 1), X(i, 2), X(i, 3), 'go', 'MarkerSize', 10, 'LineWidth', 2);
    else
        plot3(X(i, 1), X(i, 2), X(i, 3), 'mo', 'MarkerSize', 10, 'LineWidth', 2);
    end
end

% Graficar el plano de separación
[x_values, y_values] = meshgrid(-0.5:0.1:1.5, -0.5:0.1:1.5);
z_values = -(pesos(1) * x_values + pesos(2) * y_values + bias) / pesos(3);
surf(x_values, y_values, z_values, 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'FaceColor', 'cyan');

% Configurar los ejes
xlim([-0.5, 1.5]);
ylim([-0.5, 1.5]);
zlim([-0.5, 1.5]);
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Perceptrón Cubo');

grid on;
view(3); % Vista en 3D
axis equal;
hold off;
