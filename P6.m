clc;
clear;
close all;
warning off all;
% Flores Lara Alberto 5BV1

% Leer la imagen y convertirla a doble para el procesamiento
img = imread('peppers.png');
[rows, cols, ~] = size(img);

% Pedimos al usuario el número de clases
num_clases = input('¿Cuántas clases desea crear?  ');
while num_clases <= 0
    num_clases = input('Valor no permitido, intente de nuevo: ');
end

% Pedimos al usuario el número de representantes por clase
num_representantes = input('¿Cuántos representantes por clase desea? ');
while num_representantes <= 0
    num_representantes = input('Valor no permitido, intente de nuevo:  ');
end

% Calcular total de elementos
elementos = num_representantes * num_clases;

% Mostrar la imagen y permitir al usuario seleccionar áreas
figure;
imshow(img);

% Inicializar las celdas para almacenar coordenadas y valores RGB
coordenadas_x = cell(1, num_clases);
coordenadas_y = cell(1, num_clases);
colores = ['b', 'r', 'g', 'c', 'm', 'y', 'k'];  % Colores para las clases
dataset_rgb = [];
dataset_labels = [];
legendInfo = cell(1, num_clases);  % Para la leyenda

% Seleccionar áreas y generar puntos aleatorios para cada clase
for i = 1:num_clases
    fprintf('Seleccione el área para la clase %d\n', i);
    
    % El usuario selecciona un área rectangular con el ratón
    [rect, dim_rect] = imcrop(img);
    dim_rect = round(dim_rect);
    
    % Generar coordenadas aleatorias dentro del área seleccionada
    coordenadas_x{i} = randi([dim_rect(1), dim_rect(1) + dim_rect(3)], 1, num_representantes);
    coordenadas_y{i} = randi([dim_rect(2), dim_rect(2) + dim_rect(4)], 1, num_representantes);
    
    % Obtener valores RGB de los puntos aleatorios
    z = impixel(img, coordenadas_x{i}, coordenadas_y{i});
    
    % Acumular los valores RGB en el dataset
    dataset_rgb = [dataset_rgb; z];
    dataset_labels = [dataset_labels; repmat(i, num_representantes, 1)];
end

for i = 1:num_clases
    % Dibujar los puntos para la clase actual
    hold on;
    plot(coordenadas_x{i}, coordenadas_y{i}, ['o', colores(mod(i-1, length(colores))+1)], 'Markersize', 5, 'MarkerFaceColor', colores(mod(i-1, length(colores))+1));
    
    % Preparar la información para la leyenda
    legendInfo{i} = ['Clase ', num2str(i)];
end

% Añadir leyenda al gráfico
legend(legendInfo);
hold off;

seguir = true;
while seguir == true
    disp('Criterios de clasificación:')
    disp('1. Mahalanobis')
    disp('2. Distancia Euclideana')
    disp('3. Probabilidad Máxima')
    criterio = input('Seleccione el criterio: ');
    
    k_value = -1;

    if criterio == 1
        clasificador = @mahalannobis;
        classifier_name = "Mahalanobis";   
    elseif criterio == 2
        clasificador = @euclidian;
        classifier_name = "Distancia Euclidiana";
    elseif criterio == 3
        clasificador = @max_prob;
        classifier_name = "Probabilidad Máxima";
    end
    
    % RESUSTITUCIÓN
    datos_entrenamiento = elementos;
    resustitution_matrix = obtener_matriz_conf(clasificador, num_clases, dataset_rgb, dataset_labels, datos_entrenamiento, k_value);
    resustitution_accuracy = eficiencia(resustitution_matrix);

    % CROSS-VALIDATION
    iteraciones = 13;
    datos_entrenamiento = floor(elementos / 2);
    cross_val_global_matrix = zeros(num_clases, num_clases);
    for i = 1 : iteraciones
        cross_val_conf_matrix = obtener_matriz_conf(clasificador, num_clases, dataset_rgb, dataset_labels, datos_entrenamiento, k_value);
        cross_val_global_matrix = cross_val_global_matrix + cross_val_conf_matrix;
    end
    cross_val_accuracy = eficiencia(cross_val_global_matrix);

    % LEAVE ONE OUT
    loo_matrix = leave_one_out(clasificador, num_clases, dataset_rgb, dataset_labels, k_value);
    loo_accuracy = eficiencia(loo_matrix);

    % Resultados
    fprintf("Resustitución: %f\n", resustitution_accuracy);
    fprintf("Cross-Validation: %f\n", cross_val_accuracy);
    fprintf("Leave One Out: %f\n", loo_accuracy);

    % Preguntar al usuario si desea realizar nuevos cálculos
    intento = input('¿Desea intentar de nuevo? (s/n) ', 's');
    
    if ~(intento == 'S' || intento == 's')
        break;
    end
end
disp('Hasta la vista:)');

% Funciones auxiliares

% La función de eficiencia toma como argumento la matriz de confusión
function accuracy = eficiencia(conf_matrix)
    [rows, cols] = size(conf_matrix);
    predicciones = 0;
    true_positives = 0;

    for i = 1 : rows
        for j = 1 : cols
            predictions_count = conf_matrix(i, j);
            predicciones = predicciones + predictions_count;
            if i == j
                true_positives = true_positives + predictions_count;
            end
        end
    end
    accuracy = true_positives / predicciones;
end

% Obtener la matriz de confusión
function conf_matrix = obtener_matriz_conf(selected_criteria, no_clases, x, y, train_elements, k_value)
    [total_elements_count, ~] = size(y);
    if train_elements == total_elements_count
        train_data = x;
        test_data = x;
        train_labels = y;
        test_labels = y;
    else
        [train_data, train_labels, test_data, test_labels] = get_test_train_data(x, y, train_elements);
    end

    [test_elements_count, ~] = size(test_labels);
    conf_matrix = zeros(no_clases, no_clases);
    
    for element_no = 1 : test_elements_count
        vector_x = test_data(element_no, :);
        expected_output = test_labels(element_no);

        if k_value <= 0
            [predicted_class, ~] = selected_criteria(train_data, train_labels, no_clases, vector_x);
        else
            predicted_class = knn_euclidean(train_data, train_labels, k_value, vector_x);
        end

        if expected_output > 0 && expected_output <= no_clases && predicted_class > 0 && predicted_class <= no_clases
            conf_matrix(expected_output, predicted_class) = conf_matrix(expected_output, predicted_class) + 1;
        end
    end
end

% Leave One Out
function conf_matrix = leave_one_out(selected_criteria, nume_clases, X, y, k_value)
    [total_elements_count, ~] = size(y);
    conf_matrix = zeros(nume_clases, nume_clases);

    for element_no = 1:total_elements_count
        train_data = X;
        train_data(element_no, :) = [];
        
        train_labels = y;
        train_labels(element_no, :) = [];

        test_data = X(element_no, :);
        test_labels = y(element_no);

        if k_value <= 0
            [predicted_class, ~] = selected_criteria(train_data, train_labels, nume_clases, test_data);
        else
            predicted_class = knn_euclidean(train_data, train_labels, k_value, test_data);
        end
        
        if test_labels > 0 && test_labels <= nume_clases && predicted_class > 0 && predicted_class <= nume_clases
            conf_matrix(test_labels, predicted_class) = conf_matrix(test_labels, predicted_class) + 1;
        end
    end
end

% Obtener valores RGB de las coordenadas
function [dataset_rgb_values, dataset_labels] = get_rgb_from_coordinates(image, class_x_values, class_y_values, elements_p_class, class_no)
    dataset_rgb_values = zeros(elements_p_class, 3);
    dataset_labels = zeros(elements_p_class, 1);
    for i = 1 : elements_p_class
        rgb_value = image(class_y_values(i), class_x_values(i), :);
        dataset_rgb_values(i, :) = rgb_value;
        dataset_labels(i) = class_no;
    end
end

% KNN Euclidiana
function [class_no] = knn_euclidean(dataset_rgb, dataset_labels, k, vector)
    distancias = zeros(size(dataset_rgb, 1), 1);
    for i = 1:size(dataset_rgb, 1)
        distancias(i) = norm(vector - dataset_rgb(i, :));
    end
    [~, sortedIndices] = sort(distancias);
    nearestNeighbors = sortedIndices(1:k);
    neighborLabels = dataset_labels(nearestNeighbors);
    class_no = mode(neighborLabels);
end

% Clasificación por distancia Euclideana
function [class, min_dist] = euclidian(X, labels, n_classes, vector)
    distancias = zeros(1, n_classes);
    for i = 1 : n_classes
        class_data = X(labels == i, :);
        distancias(i) = mean(sqrt(sum((class_data - vector) .^ 2, 2)));
    end
    [min_dist, class] = min(distancias);
end

% Clasificación por Mahalanobis
function [class, min_dist] = mahalannobis(X, labels, n_classes, vector)
    distancias = zeros(1, n_classes);
    for i = 1 : n_classes
        class_data = X(labels == i, :);
        C = cov(class_data);
        distancias(i) = sqrt((vector - mean(class_data)) / C * (vector - mean(class_data))');
    end
    [min_dist, class] = min(distancias);
end

% Clasificación por Probabilidad Máxima
function [class, prob] = max_prob(X, labels, n_classes, vector)
    probabilidades = zeros(1, n_classes);
    for i = 1 : n_classes
        class_data = X(labels == i, :);
        probabilidades(i) = mvnpdf(vector, mean(class_data), cov(class_data));
    end
    [prob, class] = max(probabilidades);
end
% Función para dividir los datos en conjuntos de entrenamiento y prueba
function [train_data, train_labels, test_data, test_labels] = get_test_train_data(x, y, train_elements)
    total_elements = size(x, 1);
    
    % Seleccionar índices aleatorios para el conjunto de entrenamiento
    random_indices = randperm(total_elements, train_elements);
    
    % Crear el conjunto de entrenamiento
    train_data = x(random_indices, :);
    train_labels = y(random_indices);
    
    % Crear el conjunto de prueba con los elementos restantes
    test_indices = setdiff(1:total_elements, random_indices);
    test_data = x(test_indices, :);
    test_labels = y(test_indices);
end
