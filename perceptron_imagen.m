clc;
close all;
clear all;
warning off all;

img = imread("peppers.png");
figure, imshow(img);
title('Seleccione el área representativa para la Clase 1');
areaClase1 = imcrop;
title('Seleccione el área representativa para la Clase 2');
areaClase2 = imcrop;
close;

% Solicitar el número de representantes de cada clase
numRepresentantesClase1 = input('Ingrese el número de representantes para la Clase 1: ');
numRepresentantesClase2 = input('Ingrese el número de representantes para la Clase 2: ');

% Extraer valores de los canales RGB de cada área para los representantes
X_clase1 = double(reshape(areaClase1, [], 3));
X_clase2 = double(reshape(areaClase2, [], 3));

X_clase1 = X_clase1(1:numRepresentantesClase1, :);
X_clase2 = X_clase2(1:numRepresentantesClase2, :);

% Crear el conjunto de datos de entrada X y etiquetas y
X = [X_clase1; X_clase2];
y = [zeros(size(X_clase1, 1), 1); ones(size(X_clase2, 1), 1)];  % Clase 1: 0, Clase 2: 1

% Solicitar parámetros iniciales
pesos = input('Ingrese los pesos iniciales como un vector columna [w1; w2; w3]: ');
bias = input('Ingrese el valor inicial del bias: ');
learning_rate = input('Ingrese la tasa de aprendizaje inicial: ');
max_iteraciones = input('Ingrese la cantidad máxima de iteraciones: ');

itera = 0;
converged = false;

% Entrenamiento del perceptrón
while ~converged && itera < max_iteraciones
    itera = itera + 1;
    converged = true;
    pesos_previos = pesos;  % Guardar pesos para la condición de convergencia

    for i = 1:size(X, 1)
        % Calcular la salida del perceptrón
        output = pesos' * X(i, :)' + bias;
        
        % Verificar si la clasificación es incorrecta y actualizar pesos y bias
        if y(i) == 1 && output <= 0  % Clase 2 pero clasificado incorrectamente
            converged = false;
            pesos = pesos + learning_rate * X(i, :)';
            bias = bias + learning_rate;
        elseif y(i) == 0 && output >= 0  % Clase 1 pero clasificado incorrectamente
            converged = false;
            pesos = pesos - learning_rate * X(i, :)';
            bias = bias - learning_rate;
        end
    end
    
    % Verificar condición de convergencia |w(k+1) - w(k)| < 0.0001
    if norm(pesos - pesos_previos) < 0.00000000000000000001
        converged = true;
    end
end

% Mostrar el número de iteraciones y los pesos ajustados
disp(['Número final de iteraciones: ', num2str(itera)]);
disp('Pesos finales:');
disp(pesos);
disp(['Bias final: ', num2str(bias)]);

% Mostrar la función de solución
fprintf('Función de solución = %.4f*R + %.4f*G + %.4f*B + %.4f = 0\n', pesos(1), pesos(2), pesos(3), bias);

% Graficar la separación en un plano RGB si es posible
figure;
scatter3(X_clase1(:, 1), X_clase1(:, 2), X_clase1(:, 3), 'bo', 'filled');
hold on;
scatter3(X_clase2(:, 1), X_clase2(:, 2), X_clase2(:, 3), 'mo', 'filled');

% Graficar el plano de separación
[r_values, g_values] = meshgrid(0:10:255, 0:10:255);
b_values = -(pesos(1) * r_values + pesos(2) * g_values + bias) / pesos(3);
surf(r_values, g_values, b_values, 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'FaceColor', 'black');

xlabel('Rojo');
ylabel('Verde');
zlabel('Azul');
title('Clasificación de Perceptrón en Espacio de Color RGB');
grid on;
view(3);
hold off;

% Bucle para seleccionar píxeles y mostrar la clase
continuar = true;
while continuar
    figure, imshow(img);
    title('Haga clic en un píxel para clasificar o cierre la imagen para salir');
    
    % Obtener coordenadas del píxel seleccionado
    [x, y] = ginput(1);
    
    % Redondear las coordenadas para que sean índices válidos de la imagen
    x = round(x);
    y = round(y);
    
    % Verificar que las coordenadas están dentro de los límites de la imagen
    if x > 0 && x <= size(img, 2) && y > 0 && y <= size(img, 1)
        % Extraer los valores RGB del píxel seleccionado
        r = img(y, x, 1);
        g = img(y, x, 2);
        b = img(y, x, 3);
        
        % Vector del píxel seleccionado, convertido a double y en forma de columna
        pixelSeleccionado = double([r; g; b]);
        
        % Clasificar el píxel utilizando el perceptrón
        salida = pesos' * pixelSeleccionado + bias;
        
        % Mostrar la clase del píxel
        if salida >= 0
            disp('El píxel seleccionado pertenece a la Clase 2.');
        else
            disp('El píxel seleccionado pertenece a la Clase 1.');
        end
    else
        disp('Las coordenadas seleccionadas están fuera de la imagen.');
    end
    
    % Preguntar al usuario si desea continuar
    respuesta = input('¿Desea clasificar otro píxel? (s/n): ', 's');
    if lower(respuesta) ~= 's'
        continuar = false;
    end
end

function imagen = seleccionarLeerImagen()
    % Seleccionar archivo de imagen
    [archivo, ruta] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.tiff', 'Archivos de imagen (*.jpg, *.jpeg, *.png, *.bmp, *.tiff)'}, ...
                                 'Selecciona una imagen');
    % Verificar si se ha seleccionado un archivo
    if isequal(archivo, 0)
        disp('No se seleccionó ninguna imagen.');
        imagen = [];
        return;
    end

    % Leer la imagen seleccionada
    rutaCompleta = fullfile(ruta, archivo);
    imagen = imread(rutaCompleta);

    % Mostrar mensaje de éxito y la imagen
    disp(['Imagen cargada: ', rutaCompleta]);
    %imshow(imagen);
    %title('Imagen Seleccionada');
end