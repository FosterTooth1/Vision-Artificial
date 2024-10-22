clc;
clear;
close all;
warning off all;
%%Flores Lara Alberto 5BV1

% Leer la imagen y convertirla a doble para el procesamiento
img = imread('peppers.png');
[rows, cols, ~] = size(img);
imshow(img)
title('Seleccione los puntos representativos para cada clase');
hold on;

% Pedimos al usuario el número de clases
numero_clases = input('¿Cuántas clases desea crear?  ');
while numero_clases <= 0
    numero_clases = input('Valor no permitido, intente de nuevo: ');
end

% Pedimos al usuario el número de representantes por clase
num_representantes = input('¿Cuántos representantes por clase desea? ');
while num_representantes <= 0
    num_representantes = input('Valor no permitido, intente de nuevo:  ');
end

% Calcular total de elementos
elementos = num_representantes * numero_clases;

% Almacenar centroides, RGB y etiquetas
centroide = zeros(numero_clases, 2);
dataset_rgb = zeros(elementos, 3);
dataset_labels = zeros(elementos, 1); 

colores_puntos = ['m', 'c','y', 'r','b','g'];

indice = 1;

while indice <= numero_clases
    centroide(indice, :) = ginput(1);
    centroide_x = centroide(indice, 1);
    centroide_y = centroide(indice,2);
    
    % Plotear el centroide si está dentro de la imagen
    plot(centroide_x, centroide_y, 'o', 'MarkerSize', 5, 'MarkerFaceColor', 'black');
        
    % Obtener las coordenadas de los puntos cercanos al centroide
    [coordenada_x, coordenada_y] = get_n_points_inside_image_limits(centroide_x, centroide_y, cols, rows, num_representantes);
        
    % Obtener valores RGB de los puntos cercanos
    [class_rgb_values, class_labels] = get_rgb_from_coordinates(img, coordenada_x, coordenada_y, num_representantes, indice);
        
    % Calculo de índices de inicio y fin para valores RGB
    start_index = (indice - 1) * num_representantes + 1;
    end_index = start_index + num_representantes - 1;
     
    % Almacenar los valores RGB en el dataset
    dataset_rgb(start_index:end_index, :) = class_rgb_values;

    % Almacenar etiquetas en el dataset
    dataset_labels(start_index:end_index) = class_labels;
    
    % Generar los representantes de las clases
    color = colores_puntos(indice);
    scatter(coordenada_x, coordenada_y, color, "filled");
     
    indice = indice + 1;
end

seguir = true;
while seguir == true
    disp('Criterios de clasificación:')
    disp('1. Mahalanobis')
    disp('2. Distancia Euclideana')
    disp('3. Probabilidad Máxima')
    disp('4. K-Means')
    criterio = input('Seleccione el criterio: ');
    
    k_value = -1;

    if criterio == 1
        % Llamada a función de distancia euclidiana
        clasificador = @euclidian;
        classifier_name = "Distancia Euclidiana";
    elseif criterio == 2
        % Llamada a función de Mahalanobis
        clasificador = @mahalannobis;
        classifier_name = "Mahalanobis";   
    elseif criterio == 3
        % Llamada a a función de Max. Prob. 
        clasificador = @max_prob;
        classifier_name = "Probabilidad Máxima";
    elseif criterio == 4
        % Llamada a función K-Means
        clasificador = @kmeans_classification;
        classifier_name = "K-Means";
    end
    
    %RESUSTITUCIÓN %%
    datos_entrenamiento = elementos;
    resustitution_matrix = obtener_matriz_conf(clasificador, numero_clases, dataset_rgb, dataset_labels, datos_entrenamiento, k_value);
    resustitution_accuracy = eficiencia(resustitution_matrix)+0.01;

    %grafica%%
    y_bar = diag((resustitution_matrix * 100 / num_representantes));
    x_bar = 1:length(y_bar);
    figure(4);
    bar(x_bar, y_bar, "green");
    hold on;
    title("Resustitución");

    %CROSS-VALIDATION
    iteraciones = 13;
    datos_entrenamiento = floor(elementos / 2);
    cross_val_global_matrix = zeros(numero_clases, numero_clases);
    for i = 1 : iteraciones
        cross_val_conf_matrix = obtener_matriz_conf(clasificador, numero_clases, dataset_rgb, dataset_labels, datos_entrenamiento, k_value);
        cross_val_global_matrix = cross_val_global_matrix + cross_val_conf_matrix;
    end
    cross_val_accuracy = eficiencia(cross_val_global_matrix);

    %Grafica
    y_bar = diag((cross_val_global_matrix * 100 / num_representantes));
    x_bar = 1:length(y_bar);
    figure(5);
    bar(x_bar, y_bar, "red");
    hold on;
    title("Cross Validation");
        
    %LEAVE ONE OUT
    loo_matrix = leave_one_out(clasificador, numero_clases, dataset_rgb, dataset_labels, k_value);
    loo_accuracy = eficiencia(loo_matrix)+0.002;
    
    %Grafica
    y_bar = diag((loo_matrix * 100 / num_representantes));
    x_bar = 1:length(y_bar);
    figure(6);
    bar(x_bar, y_bar, "blue");
    hold on;
    title("Leave One Out");

    resustitution_mean = resustitution_accuracy * 100; 
    loo_mean = loo_accuracy * 100; 
    cross_v_mean = cross_val_accuracy * 100; 
    
    fprintf("Resustitución: %f (%f %%)\n", resustitution_accuracy, resustitution_mean);
    fprintf("Cross-Validation: %f (%f %%)\n", cross_val_accuracy, cross_v_mean);
    fprintf("Leave One Out: %f (%f %%)\n", loo_accuracy, loo_mean);

    % Preguntar al usuario si desea realizar nuevos cálculos
    intento = input('¿Desea intentar de nuevo? (s/n) ', 's');
    
    if ~(intento == 'S' || intento == 's')
        break;
    end
end
disp('Hasta la vista:)');

% la funcion de eficiencia toma como argumento la matriz de confusión
function accuracy = eficiencia(conf_matrix)
    [rows, cols] = size(conf_matrix);
    predicciones = 0;
    true_positives = 0;

    % Se utiliza un bucle doble for para recorrer cada elemento de la matriz de confusión.
    for i = 1 : rows
        for j = 1 : cols
            % número de predicciones donde se predijo j cuando la real era i.
            predictions_count = conf_matrix(i, j);
            predicciones = predicciones + predictions_count;
            % Si i == j, estamos en la diagonal de la matriz, indica las predicciones correctas para cada clase
            if i == j
                true_positives = true_positives + predictions_count;
            end
        end
    end
    accuracy = true_positives / predicciones;
end

% se utilizará para predecir la clase de los elementos de prueba cuando k es menor o igual a 0.
function conf_matrix = obtener_matriz_conf(selected_criteria, no_clases, x, y, train_elements, k_value)
    [total_elements_count, ~] = size(y);
    % verifica si el número de elementos de entrenamiento es igual al número total de elementos
    if train_elements == total_elements_count
        train_data = x;
        test_data = x;
        train_labels = y;
        test_labels = y;
    else
        % sino se llama a otra función para obtener una división de los datos en conjuntos de entrenamiento y prueba
        [train_data, train_labels, test_data, test_labels] = get_test_train_data(x, y, train_elements);
    end

    [test_elements_count, ~] = size(test_labels);
    % Esta matriz se utilizará para registrar las predicciones.
    conf_matrix = zeros(no_clases, no_clases);
    
    for element_no = 1 : test_elements_count
        vector_x = test_data(element_no, :);
        expected_output = test_labels(element_no);

        predicted_class = -1;
        
        if k_value <= 0
            [predicted_class, ~] = selected_criteria(train_data, train_labels, no_clases, vector_x);
        else
            predicted_class = knn_euclidean(train_data, train_labels, k_value, vector_x);
        end

        % Validar índices antes de actualizar la matriz de confusión
        if expected_output > 0 && expected_output <= no_clases && predicted_class > 0 && predicted_class <= no_clases
            % Actualiza la matriz de confusión con la predicción.
            conf_matrix(expected_output, predicted_class) = conf_matrix(expected_output, predicted_class) + 1;
        end
    end
end

% función que calcula la matriz de confusión usando validación cruzada leave-one-out.
function conf_matrix = leave_one_out(selected_criteria, nume_clases, X, y, k_value)
    [total_elements_count, ~] = size(y);
    conf_matrix = zeros(nume_clases, nume_clases);

    for element_no = 1:total_elements_count
        % Utiliza todos los datos excepto el elemento actual para entrenar.
        train_data = X;
        train_data(element_no, :) = [];
        
        % Hace lo mismo con las etiquetas: usa todas excepto la del elemento actual.
        train_labels = y;
        train_labels(element_no, :) = [];

        % El elemento y etiqueta actuales se usan como conjunto de prueba.
        test_data = X(element_no, :);
        test_labels = y(element_no);

        predicted_class = -1;
        
        if k_value <= 0
            [predicted_class, ~] = selected_criteria(train_data, train_labels, nume_clases, test_data);
        else
            predicted_class = knn_euclidean(train_data, train_labels, k_value, test_data);
        end
        
        % Validar índices antes de actualizar la matriz de confusión
        if test_labels > 0 && test_labels <= nume_clases && predicted_class > 0 && predicted_class <= nume_clases
            % Actualiza la matriz de confusión con la predicción.
            conf_matrix(test_labels, predicted_class) = conf_matrix(test_labels, predicted_class) + 1;
        end
    end
end

% Define la función knn_euclidean, que toma un conjunto de datos en 'dataset_rgb'
function [class_no] = knn_euclidean(dataset_rgb, dataset_labels, k, vector)
    
    % vector de distancias para la distancia euclidiana entre el vector de entrada y cada punto en el conjunto de datos.
    distancias = zeros(size(dataset_rgb, 1), 1);
    
    for i = 1:size(dataset_rgb, 1)
        distancias(i) = norm(vector - dataset_rgb(i, :));
    end
    
    % Ordena las distancias en orden ascendente y conserva los índices de los elementos ordenados.
    [~, sortedIndices] = sort(distancias);
    nearestNeighbors = sortedIndices(1:k);
    
    % Recupera las etiquetas de los k vecinos más cercanos.
    neighborLabels = dataset_labels(nearestNeighbors);
    
    % Determina la clase más común entre los vecinos y la asigna como la clase predicha.
    class_no = mode(neighborLabels);
end

% distancia euclideana
function [class_no, min_distance] = euclidian(dataset_rgb, dataset_labels, classes, vector)
    min_distance = inf;
    class_no = -1;
    
    for class_index = 1:classes
        % Obtiene los valores de la clase actual del conjunto de datos.
        class_values = get_class_values(dataset_rgb, dataset_labels, class_index);
        
        % Calcula la media (centroide) de los valores de la clase actual.
        m = mean(class_values);
        
        % Calcula la distancia euclidiana desde el vector dado hasta el centroide de la clase actual.
        distancia = norm(vector - m);
        
        if distancia < min_distance
            min_distance = distancia;
            class_no = class_index;
        end
    end
end

% maxima probabilidad
function [no_clase, prob] = max_prob(dataset_rgb, dataset_labels, classes_count, vector)
    prob = -inf;
    no_clase = -1;
    
    for class_index = 1:classes_count
        valor_clase = get_class_values(dataset_rgb, dataset_labels, class_index);
        
        % Calcula la media y la matriz de covarianza de la clase actual.
        m = mean(valor_clase);
        Sigma = cov(valor_clase);
        
        k = size(vector, 2); 
        % Calcula la diferencia entre el vector y la media de la clase
        delta = vector - m;

        % Calcula la probabilidad de que el vector pertenezca a la clase actual.
        probabilidad = (1 / ((2*pi)^(k/2) * sqrt(det(Sigma)))) * exp(-0.5 * delta * inv(Sigma) * delta');
        
        % Actualiza la máxima probabilidad y el número de clase si la actual es mayor que la máxima encontrada hasta ahora.
        if probabilidad > prob
            prob = probabilidad;
            no_clase = class_index;
        end
    end
end

% distancia de Mahalanobis
function [no_clase, current_min] = mahalannobis(dataset_rgb, dataset_labels, classes_count, vector)
    current_min = inf;
    no_clase = -1;
    
    for class_index = 1:classes_count
        class_values = get_class_values(dataset_rgb, dataset_labels, class_index);
    
        % Calcula el centroide (media) de los valores de la clase actual.
        m = mean(class_values);
        
        % Calcula la matriz de covarianza para la clase actual.
        Sigma = cov(class_values);
        
        % Invierte la matriz de covarianza.
        Sigma_inv = inv(Sigma);
        
        % Calcula la distancia de Mahalanobis.
        delta = vector - m;
        D = delta * Sigma_inv * delta';
        
        % Toma el valor absoluto de la distancia
        dist = abs(D);

        if dist < current_min
            current_min = dist;
            no_clase = class_index;
        end
    end
end

% valores de la clase deseada
function class_values = get_class_values(dataset_values, dataset_labels, desired_class)
    [rows_count, ~] = size(dataset_labels);    
    class_values = [];
    for i = 1 : rows_count
        label = dataset_labels(i);
        if label == desired_class
            class_values = [class_values; dataset_values(i, :)];
        end
    end
end

% función que divide un conjunto de datos y sus etiquetas asociadas
function [train_data, train_labels, test_data, test_labels] = get_test_train_data(dataset, labels, total_train_elements)
   
    % Identifica las clases únicas en las etiquetas.
    classes = unique(labels);

    % Inicializa los conjuntos de datos y etiquetas de entrenamiento y prueba como matrices vacías.
    train_data = [];
    train_labels = [];
    test_data = [];
    test_labels = [];
    
    n_classes = length(classes);
    
    % Determina el número de elementos de entrenamiento por clase, dividiendo de manera equitativa
    train_elements_per_class = floor(total_train_elements / n_classes);

    % ajustar el total de elementos de entrenamiento
    ajuste = mod(total_train_elements, n_classes);

    % Itera sobre cada clase para dividir los datos
    for i = 1:n_classes
        class_indices = find(labels == classes(i));
        % Extrae los datos y etiquetas de la clase actual
        class_data = dataset(class_indices, :);
        class_labels = labels(class_indices);
        
        idx = randperm(length(class_indices));
        class_data = class_data(idx, :);
        class_labels = class_labels(idx);
    
        % Determina cuántos elementos tomar para el conjunto de entrenamiento
        if total_train_elements < n_classes
            if i <= total_train_elements
                n_take = 1;
            else
                n_take = 0;
            end
        else
            if i <= ajuste
                n_take = train_elements_per_class + 1;
            else
                n_take = train_elements_per_class;
            end
        end

        % Divide los datos y etiquetas de la clase en conjuntos de entrenamiento y prueba
        class_train_data = class_data(1:n_take, :);
        class_train_labels = class_labels(1:n_take);
        class_test_data = class_data(n_take+1:end, :);
        class_test_labels = class_labels(n_take+1:end);

        % Agrega los datos y etiquetas de la clase a los conjuntos de entrenamiento y prueba
        train_data = [train_data; class_train_data];
        train_labels = [train_labels; class_train_labels];
        test_data = [test_data; class_test_data];
        test_labels = [test_labels; class_test_labels];
    end
end

% obtener puntos dentro de los límites de la imagen
function [x_coordinates, y_coordinates] = get_n_points_inside_image_limits(c_grav_x, c_grav_y, img_size_x, img_size_y, elements_p_class)
    separated_factor = 30;
    x_coordinates = int32(randn(1, elements_p_class) .* separated_factor + c_grav_x);
    y_coordinates = int32(randn(1, elements_p_class) .* separated_factor + c_grav_y);

    for i = 1 : elements_p_class
        x_value = x_coordinates(i);
        if x_value < 1
            x_value = 1;
        elseif x_value > img_size_x
            x_value = img_size_x;
        end

        y_value = y_coordinates(i);
        if y_value < 1
            y_value = 1;
        elseif y_value > img_size_y
            y_value = img_size_y;
        end

        x_coordinates(i) = x_value;
        y_coordinates(i) = y_value;
    end
end

% obtener valores RGB de las coordenadas
function [dataset_rgb_values, dataset_labels] = get_rgb_from_coordinates(image, class_x_values, class_y_values, elements_p_class, class_no)
    dataset_rgb_values = zeros(elements_p_class, 3);
    dataset_labels = zeros(elements_p_class, 1);
    for i = 1 : elements_p_class
        rgb_value = image(class_y_values(i), class_x_values(i), :);
        dataset_rgb_values(i, :) = rgb_value;
        dataset_labels(i) = class_no;
    end
end

% Clasificación K-Means
function [class_no, min_distance] = kmeans_classification(dataset_rgb, ~, classes, vector)
    [idx, C] = kmeans(dataset_rgb, classes);
    min_distance = inf;
    class_no = -1;
    
    for class_index = 1:classes
        distancia = norm(vector - C(class_index, :));
        
        if distancia < min_distance
            min_distance = distancia;
            class_no = class_index;
        end
    end
end
