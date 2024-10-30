clc;
close all;
clear;
warning off;

colors = ['y', 'm', 'c', 'w', 'k'];

msj_no_classes = 'Ingresa el número de clases a utilizar: ';
no_classes = input(msj_no_classes);

msj_no_elementos_p_clase = 'Ingresa el número de elementos por clase: ';
no_elementos = input(msj_no_elementos_p_clase);

% Número total de elementos en el dataset
no_elementos_dataset = no_elementos * no_classes;

% Inicialización de matrices para almacenar datos
classes_elements = zeros(no_elementos_dataset, 3);

playa = imread('peppers.png');

[rows, cols, ~] = size(playa);

imshow(playa);
hold on;

% Matrices para el centro de gravedad y los datos RGB
c_grav = zeros(length(no_classes), 2);
dataset_rgb = zeros(no_elementos_dataset, 3);
dataset_labels = zeros(no_elementos_dataset, 1); %-> Clases
counter = 1;

while counter <= no_classes
    fprintf('Selecciona una ventana para la clase %d:\n', counter);
    
    % Seleccionar la ventana usando rbbox
    rect = getrect;
    x_min = max(1, floor(rect(1)));
    y_min = max(1, floor(rect(2)));
    width = min(cols - x_min, floor(rect(3)));
    height = min(rows - y_min, floor(rect(4)));
    
    % Validar si la selección es válida
    if width <= 0 || height <= 0
        fprintf('\n\nSelecciona solo ventanas válidas dentro de la imagen\n');
        continue;
    end
    
    % Mostrar la ventana seleccionada
    rectangle('Position', [x_min, y_min, width, height], 'EdgeColor', 'r');
    
    % Obtener puntos aleatorios dentro de la ventana seleccionada
    [x_coordinates, y_coordinates] = get_random_points_within_window(x_min, y_min, width, height, no_elementos);
    
    % Obtener los valores RGB de los puntos seleccionados
    [class_rgb_values, class_labels] = get_rgb_from_coordinates(playa, x_coordinates, y_coordinates, no_elementos, counter);
    
    start_idx = (counter - 1) * no_elementos + 1;
    end_idx = start_idx + no_elementos - 1;

    % Crear dataset de valores RGB y etiquetas
    dataset_rgb(start_idx:end_idx, :) = class_rgb_values;
    dataset_labels(start_idx:end_idx) = class_labels; %->Indica de qué clase pertenecen los datos
    
    % Ploteamos los puntos
    color = colors(counter);
    scatter(x_coordinates, y_coordinates, color, 'filled');
    
    counter = counter + 1;
end

% Función para obtener puntos aleatorios dentro de una ventana seleccionada
function [x_coordinates, y_coordinates] = get_random_points_within_window(x_min, y_min, width, height, elements_p_class)
    x_coordinates = randi([x_min, x_min + width], 1, elements_p_class);
    y_coordinates = randi([y_min, y_min + height], 1, elements_p_class);
end

user_input = 'c';


%Comenzamos con los criterios yeah
while strcmp(user_input, 'c')
    msg_selected_criteria = 'Ingresa el número del criterio para la clasificación: ';
    fprintf("\n1. Mahalanobis \n2. Distancia Euclidiana \n3. Máxima Probabilidad \n4. Ejecutar todas las métricas\n");
    selected_criteria_idx = input(msg_selected_criteria);
    k_for_knn = -1;

    % Identificadores para funciones y nombres de cada criterio
    criteria_functions = {@get_class_and_min_dist_mahalannobis, @get_class_and_min_dist_euclidean, @get_class_and_max_prob};
    criteria_names = ["Mahalanobis", "Distancia Euclidiana", "Máxima Probabilidad"];
    avg_accuracies = zeros(1, 3);

    if selected_criteria_idx == 4
        % Ejecutar todas las métricas y almacenar precisión promedio para cada criterio
        for i = 1:3
            selected_criteria_function = criteria_functions{i};
            selected_criteria_name = criteria_names(i);

            fprintf("\nUsando %s\n", selected_criteria_name);

            % Resustitución
            disp("RESUSTITUCIÓN (TODOS CONTRA TODOS)")
            total_train_elements = no_elementos_dataset;
            resustitution_conf_matrix = get_conf_matrix_using_f(selected_criteria_function, no_classes, dataset_rgb, dataset_labels, total_train_elements, k_for_knn);
            resustitution_accuracy = get_accuracy(resustitution_conf_matrix);

            % Cross-validation 50/50, 20 iteraciones
            iterations = 20;
            fprintf("CROSS-VALIDATION 50/50, %d iteraciones\n", iterations);
            total_train_elements = floor(no_elementos_dataset / 2);
            cross_val_global_conf_matrix = zeros(no_classes, no_classes);
            for j = 1:iterations
                cross_val_conf_matrix = get_conf_matrix_using_f(selected_criteria_function, no_classes, dataset_rgb, dataset_labels, total_train_elements, k_for_knn);
                cross_val_global_conf_matrix = cross_val_global_conf_matrix + cross_val_conf_matrix;
            end
            cross_val_accuracy = get_accuracy(cross_val_global_conf_matrix);

            % Leave-one-out
            disp("LEAVE ONE OUT")
            leave_one_out_conf_matrix = leave_one_out_using_f(selected_criteria_function, no_classes, dataset_rgb, dataset_labels, k_for_knn);
            leave_one_out_accuracy = get_accuracy(leave_one_out_conf_matrix);

            % Calcular precisión promedio
            avg_accuracies(i) = mean([resustitution_accuracy, cross_val_accuracy, leave_one_out_accuracy]);
            % Mostrar precisión promedio en consola
            fprintf("Precisión promedio para %s: %f\n", selected_criteria_name, avg_accuracies(i));
        end

        % Graficar comparación de precisión promedio entre criterios
        figure;
        bar(categorical(criteria_names), avg_accuracies, 'cyan');
        title('Comparación de Precisión Promedio entre Métodos de Clasificación');
        ylabel('Precisión Promedio');
        xlabel('Método de Clasificación');
     

    else
        % Selección de criterio individual como en el código original
        if selected_criteria_idx == 1
            selected_criteria_function = @get_class_and_min_dist_mahalannobis;
            selected_criteria_name = "Distancia Mahalanobis";
        elseif selected_criteria_idx == 2
            selected_criteria_function = @get_class_and_min_dist_euclidean;
            selected_criteria_name = "Distancia Euclidiana";
        elseif selected_criteria_idx == 3
            selected_criteria_function = @get_class_and_max_prob;
            selected_criteria_name = "Criterio de Máxima Probabilidad";
        end

        % Ejecutar las métricas de evaluación de acuerdo al criterio seleccionado
        fprintf("\nUsando %s\n", selected_criteria_name);
        %% Resustitución %%
        disp("RESUSTITUCIÓN (TODOS CONTRA TODOS)")
        total_train_elements = no_elementos_dataset;
        resustitution_conf_matrix = get_conf_matrix_using_f(selected_criteria_function, no_classes, dataset_rgb, dataset_labels, total_train_elements, k_for_knn);
        resustitution_accuracy = get_accuracy(resustitution_conf_matrix);

        %% Cross-Validation 50/50, 20 iteraciones %%
        iterations = 20;
        fprintf("CROSS-VALIDATION 50/50, %d iteraciones\n", iterations);
        total_train_elements = floor(no_elementos_dataset / 2);
        cross_val_global_conf_matrix = zeros(no_classes, no_classes);
        for i = 1:iterations
            cross_val_conf_matrix = get_conf_matrix_using_f(selected_criteria_function, no_classes, dataset_rgb, dataset_labels, total_train_elements, k_for_knn);
            cross_val_global_conf_matrix = cross_val_global_conf_matrix + cross_val_conf_matrix;
        end
        cross_val_accuracy = get_accuracy(cross_val_global_conf_matrix);

        %% Leave-One-Out %%
        disp("LEAVE ONE OUT")
        leave_one_out_conf_matrix = leave_one_out_using_f(selected_criteria_function, no_classes, dataset_rgb, dataset_labels, k_for_knn);
        leave_one_out_accuracy = get_accuracy(leave_one_out_conf_matrix);

        % Mostrar los resultados de precisión
        fprintf("ACCURACY USANDO %s\n", selected_criteria_name);
        fprintf("RESUSTITUCIÓN: %f\n", resustitution_accuracy);
        fprintf("CROSS-VALIDATION 50/50 (%d ITERACIONES): %f\n", iterations, cross_val_accuracy);
        fprintf("LEAVE ONE OUT: %f\n", leave_one_out_accuracy);

        % Graficar precisión de métricas individuales
        x_acc = ["Resustitución", "CrossValidation", "Leave One Out"];
        y_acc = [resustitution_accuracy, cross_val_accuracy, leave_one_out_accuracy];
        figure;
        bar(x_acc, y_acc, 'yellow');
        hold on;
        title("Precisión Usando " + selected_criteria_name);
        ylabel('Precisión');
        xlabel('Métrica de Validación');
    end

    disp('¿Deseas usar otro criterio de clasificación? c: Continuar. Cualquier otra tecla: Salir');
    user_input = input('Teclea la opción deseada: ', 's');
end

%esto ps el accuracy 
function accuracy = get_accuracy(conf_matrix)
    [rows, cols] = size(conf_matrix);
    total_predictions = 0;
    true_positives = 0;

    for i = 1 : rows
        for j = 1 : cols
            predictions_count = conf_matrix(i, j);
            total_predictions = total_predictions + predictions_count;
            if i == j
                true_positives = true_positives + predictions_count;
            end
        end
    end

    accuracy = true_positives / total_predictions;
end

%%Aqui ocurre la magia x2
%Esta misma se usa para cross y resustitucion pq solo cambia el n valores
function conf_matrix = get_conf_matrix_using_f(selected_criteria_function, no_classes, X, y, total_train_elements, k_for_knn)
    [total_elements_count, ~] = size(y);
    if total_train_elements == total_elements_count
        train_data = X;
        test_data = X;
        train_labels = y;
        test_labels = y;
    else
        [train_data, train_labels, test_data, test_labels] = get_test_train_data(X, y, total_train_elements);
    end

    [test_elements_count, ~] = size(test_labels);
    conf_matrix = zeros(no_classes, no_classes);
    
    for element_no = 1 : test_elements_count
        vector_x = test_data(element_no, :);
        expected_output = test_labels(element_no);

        predicted_class = -1;
        
        if k_for_knn <= 0
            [predicted_class, ~] = selected_criteria_function(train_data, train_labels, no_classes, vector_x);
        else
            predicted_class = knn_euclidean(train_data, train_labels, k_for_knn, vector_x);
        end

        conf_matrix(expected_output, predicted_class) = conf_matrix(expected_output, predicted_class) + 1;
    end
end

function conf_matrix = leave_one_out_using_f(selected_criteria_function, no_classes, X, y, k_for_knn)
    [total_elements_count, ~] = size(y);
    conf_matrix = zeros(no_classes, no_classes);

    for element_no = 1:total_elements_count
        % Usa todos los elementos, excepto el actual en "element_no"
        train_data = X;
        train_data(element_no, :) = [];
        
        train_labels = y;
        train_labels(element_no, :) = [];
        
        % Usa el elemento excluido de prueba
        test_data = X(element_no, :);
        test_labels = y(element_no);

        predicted_class = -1;
        
        if k_for_knn <= 0
            [predicted_class, ~] = selected_criteria_function(train_data, train_labels, no_classes, test_data);
        else
            predicted_class = knn_euclidean(train_data, train_labels, k_for_knn, test_data);
        end
        %El +1 al final es para contrarrestar el +1 de abajo usado
        %para garantizar que se dividen de manera equitativa
        conf_matrix(test_labels, predicted_class) = conf_matrix(test_labels, predicted_class) + 1;
    end
end

%Knn 
function [class_no] = knn_euclidean(dataset_rgb, dataset_labels, k, vector)
    distances = zeros(size(dataset_rgb, 1), 1);
    
    for i = 1:size(dataset_rgb, 1)
        distances(i) = norm(vector - dataset_rgb(i, :));
    end
    
    [~, sortedIndices] = sort(distances);
    nearestNeighbors = sortedIndices(1:k);
    
    neighborLabels = dataset_labels(nearestNeighbors);
    
    class_no = mode(neighborLabels);
end

%Euclideana
function [class_no, min_distance] = get_class_and_min_dist_euclidean(dataset_rgb, dataset_labels, classes_count, vector)
    min_distance = inf;
    class_no = -1;
    
    for class_index = 1:classes_count
        class_values = get_class_values(dataset_rgb, dataset_labels, class_index);
        
        mu = mean(class_values);
        
        distance = norm(vector - mu);
        
        if distance < min_distance
            min_distance = distance;
            class_no = class_index;
        end
    end
end

%Max Prob
function [class_no, max_likelihood] = get_class_and_max_prob(dataset_rgb, dataset_labels, classes_count, vector)
    max_likelihood = -inf;
    class_no = -1;
    
    for class_index = 1:classes_count
        class_values = get_class_values(dataset_rgb, dataset_labels, class_index);
        
        mu = mean(class_values);
        Sigma = cov(class_values);
        
        k = size(vector, 2); 
        delta = vector - mu;
        likelihood = (1 / ((2*pi)^(k/2) * sqrt(det(Sigma)))) * exp(-0.5 * delta * inv(Sigma) * delta');
        
        if likelihood > max_likelihood
            max_likelihood = likelihood;
            class_no = class_index;
        end
    end
end

%Mahalanobizzzzzzz
function [class_no, current_min] = get_class_and_min_dist_mahalannobis(dataset_rgb, dataset_labels, classes_count, vector)
    current_min = inf;
    class_no = -1;
    
    for class_index = 1:classes_count
        class_values = get_class_values(dataset_rgb, dataset_labels, class_index);
    
        mu = mean(class_values);
        
        Sigma = cov(class_values);
        
        Sigma_inv = inv(Sigma);
        
        delta = vector - mu;
        D2 = delta * Sigma_inv * delta';
        
        dist = abs(D2);

   
        if dist < current_min
            current_min = dist;
            class_no = class_index;
        end
        
    end
end


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

%%AQUI OCURRE LA MAGIA %%
function [train_data, train_labels, test_data, test_labels] = get_test_train_data(dataset, labels, total_train_elements)
   
%Como en labels se pueden repetir, esto identifica cuantas clases hay
    classes = unique(labels);

    train_data = [];
    train_labels = [];
    test_data = [];
    test_labels = [];
    
    n_classes = length(classes);
    %Divide el numero de elementos totales entre el numero de clases
    %Que son el numero de datos que debe asignar a cada clase
    train_elements_per_class = floor(total_train_elements / n_classes);

    %Residuo que se usa para el leave one out por ejemplo, cual sobra pues
    remainder = mod(total_train_elements, n_classes);

    %Para cada clase unica realiza un ciclo
    for i = 1:n_classes
        class_indices = find(labels == classes(i));
        class_data = dataset(class_indices, :);
        class_labels = labels(class_indices);
        
        %Mezcla random para los datos de prueba
        idx = randperm(length(class_indices));
        class_data = class_data(idx, :);
        class_labels = class_labels(idx);
    

        %Todo lo de abajo es para elegir los elementos de datos y
        %labels de entrenamiento
        if total_train_elements < n_classes
            if i <= total_train_elements
                n_take = 1;
            else
                n_take = 0;
            end
        else
            if i <= remainder
                n_take = train_elements_per_class + 1; %recuerda este +1
            else
                n_take = train_elements_per_class;
            end
        end



        class_train_data = class_data(1:n_take, :);
        class_train_labels = class_labels(1:n_take);
        class_test_data = class_data(n_take+1:end, :);
        class_test_labels = class_labels(n_take+1:end);

        train_data = [train_data; class_train_data];
        train_labels = [train_labels; class_train_labels];
        test_data = [test_data; class_test_data];
        test_labels = [test_labels; class_test_labels];
    end
end

%Plotear nubecita de random points
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

%Obtenemos el RGB de cada punto random generado, regresando una matriz rgb
%y de labels
function [dataset_rgb_values, dataset_labels] = get_rgb_from_coordinates(image, class_x_values, class_y_values, elements_p_class, class_no)
    dataset_rgb_values = zeros(elements_p_class, 3);
    dataset_labels = zeros(elements_p_class, 1);
    for i = 1 : elements_p_class
        rgb_value = image(class_y_values(i), class_x_values(i), :);
        dataset_rgb_values(i, :) = rgb_value;
        dataset_labels(i) = class_no;
    end
end

%Funcion para checar si da click dentro de la imagen, recibe cg_x, cg_y, y
%tamaño de la imagen filas y columnas.
function point_in_image = point_is_in_image(x, y, img_size_x, img_size_y)
    if  (x >= 1 && y >= 1) && (x <= img_size_x && y <= img_size_y)
        point_in_image = true;
        return
    end

    point_in_image = false;
end