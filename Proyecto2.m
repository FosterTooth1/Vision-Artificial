close all;
clear;
clc;

% Definir la carpeta que contiene las imágenes
folderPath = 'C:\Users\albsa\Desktop\Vision Artificial\DBVision';

% Verificar si la carpeta existe
if ~isfolder(folderPath)
    error('La carpeta especificada no existe: %s', folderPath);
end

% Obtener la lista de archivos de imagen en la carpeta
imageExtensions = {'*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff'};
imageFiles = [];
for i = 1:length(imageExtensions)
    imageFiles = [imageFiles; dir(fullfile(folderPath, imageExtensions{i}))];
end

% Verificar si hay imágenes en la carpeta
if isempty(imageFiles)
    error('No se encontraron imágenes en la carpeta: %s', folderPath);
end

% Definir el tamaño mínimo de área para considerar un objeto
MinArea = 500; % Número mínimo de píxeles

% Inicializar una tabla para almacenar los resultados de todas las imágenes
Resultados_Totales = table();

% Crear carpetas para guardar las imágenes modificadas y tablas de características
modifiedFolder = fullfile(folderPath, 'Imagenes_Modificadas');
featuresFolder = fullfile(folderPath, 'Caracteristicas');
if ~exist(modifiedFolder, 'dir')
    mkdir(modifiedFolder);
end
if ~exist(featuresFolder, 'dir')
    mkdir(featuresFolder);
end

% Iterar sobre cada imagen en la carpeta
for i = 1:length(imageFiles)
    % Obtener el nombre completo del archivo
    filename = imageFiles(i).name;
    filepath = fullfile(folderPath, filename);

    fprintf('Procesando imagen %d de %d: %s\n', i, length(imageFiles), filename);

    % Leer la imagen
    img = imread(filepath);

    % Verificar si la imagen es RGB
    if size(img, 3) ~= 3
        warning('La imagen %s no es RGB. Se omitirá.', filename);
        continue; % Saltar a la siguiente imagen
    end

    % Identificar los píxeles que son puramente blancos (R=255, G=255, B=255)
    mask_white = all(img == 255, 3); % Máscara lógica para píxeles blancos

    % Crear una copia de la imagen para modificar
    img_modified = img;

    % Convertir los píxeles blancos a negros
    img_modified(repmat(mask_white, [1 1 3])) = 0;

    % Crear una máscara de objetos: píxeles que no son completamente negros
    mask_objects = any(img_modified ~= 0, 3); % Resultado lógico

    % Convertir la máscara a una imagen binaria
    binary_image = mask_objects;

    % Rellenar huecos dentro de los objetos para una mejor detección
    binary_image = imfill(binary_image, 'holes');

    % Aplicar el filtrado para eliminar objetos pequeños
    binary_filtered = bwareaopen(binary_image, MinArea);

    % Etiquetar los componentes conectados
    cc = bwconncomp(binary_filtered);

    % Contar el número de objetos
    num_objects = cc.NumObjects;

    fprintf('Número de objetos detectados (≥ %d píxeles): %d\n', MinArea, num_objects);

    % Si no se detectaron objetos, continuar con la siguiente imagen
    if num_objects == 0
        warning('No se detectaron objetos en la imagen %s.', filename);
        continue;
    end

    %% Cálculo de Características para Cada Objeto

    % Inicializar arreglos para almacenar las características
    Objeto_Num = (1:num_objects)';
    Cantidad_Pixeles = zeros(num_objects,1);
    Promedio_R = zeros(num_objects,1);
    Promedio_G = zeros(num_objects,1);
    Promedio_B = zeros(num_objects,1);
    Promedio_RGB = zeros(num_objects,1);

    % Extraer los canales R, G, B de la imagen modificada
    R_channel = double(img_modified(:,:,1));
    G_channel = double(img_modified(:,:,2));
    B_channel = double(img_modified(:,:,3));

    % Iterar sobre cada objeto para calcular las características
    for j = 1:num_objects
        % Obtener los índices de píxeles del objeto actual
        idx = cc.PixelIdxList{j};

        % Cantidad de píxeles
        Cantidad_Pixeles(j) = length(idx);

        % Promedio de cada canal de color
        Promedio_R(j) = mean(R_channel(idx));
        Promedio_G(j) = mean(G_channel(idx));
        Promedio_B(j) = mean(B_channel(idx));

        % Promedio RGB combinado
        Promedio_RGB(j) = (Promedio_R(j) + Promedio_G(j) + Promedio_B(j)) / 3;
    end

    %% Crear una Tabla con las Características de la Imagen Actual

    Caracteristicas_Imagen = table(Objeto_Num, Cantidad_Pixeles, Promedio_R, Promedio_G, Promedio_B, Promedio_RGB, ...
                                   'VariableNames', {'Objeto', 'Cantidad_Pixeles', 'Promedio_R', 'Promedio_G', 'Promedio_B', 'Promedio_RGB'});

    % Añadir una columna con el nombre de la imagen
    Caracteristicas_Imagen.Nombre_Imagen = repmat({filename}, num_objects, 1);

    % Reordenar las columnas para que el nombre de la imagen esté al inicio
    Caracteristicas_Imagen = movevars(Caracteristicas_Imagen, 'Nombre_Imagen', 'Before', 'Objeto');

    % Añadir las características de la imagen actual a la tabla total
    Resultados_Totales = [Resultados_Totales; Caracteristicas_Imagen];

    %% Visualización y Guardado de Resultados para la Imagen Actual

    % Crear una figura con una tabla visual para la imagen actual
    f = figure('Name', ['Características de ', filename], 'Position', [100 100 800 200]);
    try
        uitable(f, 'Data', Caracteristicas_Imagen{:,2:end}, ... % Excluir 'Nombre_Imagen'
                    'ColumnName', Caracteristicas_Imagen.Properties.VariableNames(2:end), ...
                    'RowName', [], ...
                    'Units', 'Normalized', ...
                    'Position', [0 0 1 1]);
    catch ME
        disp(['Error al crear la tabla para la imagen ', filename, ': ', ME.message]);
    end

    % Opcional: Guardar la tabla de características como un archivo .csv
    csvFilename = fullfile(featuresFolder, [filename, '_Caracteristicas.csv']);
    writetable(Caracteristicas_Imagen, csvFilename);

    % Opcional: Guardar la imagen modificada en la carpeta de imágenes modificadas
    modifiedFilename = fullfile(modifiedFolder, ['Modificado_', filename]);
    imwrite(img_modified, modifiedFilename);

    % Cerrar la figura de la tabla después de un breve retraso para evitar demasiadas ventanas abiertas
    pause(0.5); % Ajustar el tiempo según sea necesario
    close(f);
end

%% Visualización de Resultados Totales

% Mostrar la tabla consolidada en la consola
disp('Resultados Consolidados de Todas las Imágenes:');
disp(Resultados_Totales);

% Guardar la tabla consolidada como un archivo .csv
consolidatedCSV = fullfile(featuresFolder, 'Resultados_Totales.csv');
writetable(Resultados_Totales, consolidatedCSV);

% Crear una figura con la tabla consolidada visual
f_total = figure('Name', 'Resultados Consolidados de Todas las Imágenes', 'Position', [150 150 1000 600]);
try
    uitable(f_total, 'Data', table2cell(Resultados_Totales), ... % Convertir a celda para soportar tipos mixtos
                'ColumnName', Resultados_Totales.Properties.VariableNames, ...
                'RowName', [], ...
                'Units', 'Normalized', ...
                'Position', [0 0 1 1]);
catch ME
    disp(['Error al crear la tabla consolidada: ', ME.message]);
end

% Cerrar la figura de resultados totales después de un breve retraso
pause(1);
close(f_total);

fprintf('Procesamiento completado. Los resultados se han guardado en:\n%s\n', featuresFolder);

%% --- Inicio de la Nueva Funcionalidad: Clasificador k-means ---

% Paso 1: Preparar los Datos para k-means
% Seleccionar las características para la clasificación
% En este ejemplo, usaremos Promedio_R, Promedio_G, Promedio_B y Promedio_RGB
% Puedes ajustar las características según tus necesidades

features = Resultados_Totales{:, {'Promedio_R', 'Promedio_G', 'Promedio_B', 'Promedio_RGB'}};

% Opcional: Normalizar las características para que todas tengan igual peso
features_normalized = normalize(features);

% Paso 2: Aplicar k-means para Clasificar los Objetos
k = 5; % Número de clases
rng(1); % Para reproducibilidad
[idx, C] = kmeans(features_normalized, k, 'Replicates', 5);

% Añadir los índices de cluster a la tabla
Resultados_Totales.Cluster = idx;

% Paso 3: Identificar Imágenes Representativas
% Buscamos la primera imagen que contenga únicamente objetos de cada clase

% Inicializar una celda para almacenar las etiquetas de cada clase
etiquetas_clases = cell(k,1);

for class_num = 1:k
    % Obtener los nombres únicos de las imágenes
    images = unique(Resultados_Totales.Nombre_Imagen);
    
    % Iterar sobre cada imagen para encontrar una que contenga solo objetos de clase 'class_num'
    imagen_encontrada = false;
    for j = 1:length(images)
        filename = images{j};
        datos_imagen = Resultados_Totales(strcmp(Resultados_Totales.Nombre_Imagen, filename), :);
        
        % Verificar si todos los objetos en esta imagen pertenecen a 'class_num'
        if all(datos_imagen.Cluster == class_num)
            % Si es así, esta es la imagen representativa para la clase 'class_num'
            filepath = fullfile(folderPath, filename);
            img = imread(filepath);
            
            % Mostrar la imagen
            figure('Name', ['Clase ', num2str(class_num), ' - ', filename], 'NumberTitle', 'off');
            imshow(img);
            title(['Clase ', num2str(class_num), ' - ', filename]);
            
            % Pedir al usuario que ingrese la etiqueta para esta clase
            prompt = sprintf('¿Qué etiqueta recibe la Clase %d?', class_num);
            dlg_title = sprintf('Asignar Etiqueta a la Clase %d', class_num);
            num_lines = 1;
            defaultans = {sprintf('Clase_%d', class_num)};
            answer = inputdlg(prompt, dlg_title, num_lines, defaultans);
            
            % Validar la entrada del usuario
            if isempty(answer)
                error('El usuario canceló la asignación de etiquetas.');
            end
            etiquetas_clases{class_num} = answer{1};
            
            % Cerrar la figura de la imagen
            close(gcf);
            
            imagen_encontrada = true;
            break; % Salir del bucle de imágenes para pasar a la siguiente clase
        end
    end
    
    % Si no se encontró una imagen que contenga solo objetos de 'class_num'
    if ~imagen_encontrada
        warning('No se encontró una imagen con únicamente objetos de la Clase %d.', class_num);
        % Asignar una etiqueta por defecto o permitir que el usuario la ingrese manualmente
        prompt = sprintf('Ingrese una etiqueta para la Clase %d (Asignación por defecto: Clase_%d):', class_num, class_num);
        dlg_title = sprintf('Asignar Etiqueta a la Clase %d', class_num);
        num_lines = 1;
        defaultans = {sprintf('Clase_%d', class_num)};
        answer = inputdlg(prompt, dlg_title, num_lines, defaultans);
        
        if isempty(answer)
            etiquetas_clases{class_num} = sprintf('Clase_%d', class_num);
            disp(['Se asignó la etiqueta por defecto: Clase_', num2str(class_num)]);
        else
            etiquetas_clases{class_num} = answer{1};
        end
    end
end

% Paso 4: Asignar Etiquetas a Todos los Objetos y Actualizar la Tabla

% Crear una columna de etiquetas basada en los clusters
Resultados_Totales.Etiqueta = etiquetas_clases(Resultados_Totales.Cluster);

% Guardar la tabla actualizada
updatedCSV = fullfile(featuresFolder, 'Resultados_Totales_Con_Etiquetas.csv');
writetable(Resultados_Totales, updatedCSV);

% Mostrar una confirmación
disp('Las etiquetas han sido asignadas y la tabla ha sido actualizada.');

% Opcional: Guardar la tabla consolidada visual con etiquetas
f_total_updated = figure('Name', 'Resultados Consolidados con Etiquetas', 'Position', [150 150 1200 800]);
try
    uitable(f_total_updated, 'Data', table2cell(Resultados_Totales), ... % Convertir a celda para soportar tipos mixtos
                'ColumnName', Resultados_Totales.Properties.VariableNames, ...
                'RowName', [], ...
                'Units', 'Normalized', ...
                'Position', [0 0 1 1]);
catch ME
    disp(['Error al crear la tabla consolidada con etiquetas: ', ME.message]);
end

% Guardar la figura como una imagen (PNG)
figureFile_updated = fullfile(featuresFolder, 'Resultados_Totales_Con_Etiquetas.png');
saveas(f_total_updated, figureFile_updated);

% Cerrar la figura de resultados totales actualizada después de un breve retraso
pause(1);
close(f_total_updated);

fprintf('Proceso de clasificación y etiquetado completado. Los resultados actualizados se han guardado en:\n%s\n', featuresFolder);

%% --- Inicio de la Nueva Funcionalidad: Memorama ---

% Preparar los datos necesarios para el memorama
% Seleccionar las 25 imágenes aleatorias y crear el arreglo de 50 elementos

% Obtener una lista única de imágenes
uniqueImages = unique(Resultados_Totales.Nombre_Imagen);

% Verificar que hay al menos 25 imágenes para seleccionar
if length(uniqueImages) < 25
    error('No hay suficientes imágenes únicas para crear el memorama. Se requieren al menos 25 imágenes únicas.');
end

% Seleccionar 25 imágenes al azar
selectedImages = uniqueImages(randperm(length(uniqueImages), 25));

% Crear el arreglo de 50 elementos con pares
memoramaArray = [selectedImages; selectedImages];
memoramaArray = memoramaArray(:)'; % Convertir a un vector fila de 1x50

% Revolver el arreglo
memoramaArray = memoramaArray(randperm(length(memoramaArray)));

% Inicializar variables para el juego
matchedIndices = false(1,50); % Indicadores de pares encontrados
playerScores = [0, 0]; % [Jugador1, Jugador2]
currentPlayer = 1; % 1 para Jugador 1, 2 para Jugador 2

fprintf('\n--- Bienvenido al Memorama ---\n');

% Iniciar el juego
while any(~matchedIndices)
    % Mostrar el tablero actual
    displayGrid(memoramaArray, matchedIndices);
    
    % Mostrar puntajes actuales
    fprintf('Puntajes:\nJugador 1: %d\nJugador 2: %d\n', playerScores(1), playerScores(2));
    
    % Solicitar selección de dos números al jugador actual
    fprintf('\nTurno del Jugador %d\n', currentPlayer);
    [idx1, idx2] = getPlayerSelection(memoramaArray, matchedIndices);
    
    % Obtener los nombres de las imágenes seleccionadas
    img1 = memoramaArray{idx1};
    img2 = memoramaArray{idx2};
    
    % Mostrar las imágenes seleccionadas
    displaySelectedImages(folderPath, img1, img2, idx1, idx2);
    
    % Obtener las características relevantes para comparar
    % Por ejemplo, contar la cantidad de objetos por clase en cada imagen
    counts1 = getImageClassCounts(Resultados_Totales, img1);
    counts2 = getImageClassCounts(Resultados_Totales, img2);
    
    fprintf('\nImagen %d: %s\n', idx1, img1);
    disp(counts1);
    fprintf('Imagen %d: %s\n', idx2, img2);
    disp(counts2);
    
    % Verificar si las dos imágenes son iguales en términos de objetos por clase
    if isequal(counts1, counts2)
        fprintf('¡Pareja encontrada!\n');
        matchedIndices([idx1, idx2]) = true;
        playerScores(currentPlayer) = playerScores(currentPlayer) + 1;
    else
        fprintf('No es una pareja. Cambia el turno.\n');
        % Pausar para que el jugador pueda ver las imágenes antes de continuar
        pause(2);
        % Cambiar de jugador
        currentPlayer = 3 - currentPlayer; % Cambia entre 1 y 2
    end
end

% Mostrar el tablero final
displayGrid(memoramaArray, matchedIndices);

% Mostrar puntajes finales
fprintf('\n--- Juego Finalizado ---\n');
fprintf('Puntajes Finales:\nJugador 1: %d\nJugador 2: %d\n', playerScores(1), playerScores(2));

if playerScores(1) > playerScores(2)
    fprintf('¡Jugador 1 gana!\n');
elseif playerScores(2) > playerScores(1)
    fprintf('¡Jugador 2 gana!\n');
else
    fprintf('¡Es un empate!\n');
end

%% --- Definición de Funciones ---

function displayGrid(memoramaArray, matchedIndices)
    % Función para mostrar el tablero del memorama en la terminal
    fprintf('\nTablero Actual:\n');
    for i = 1:50
        if matchedIndices(i)
            fprintf('[X] ');
        else
            fprintf('[%d] ', i);
        end
        if mod(i,10) == 0
            fprintf('\n');
        end
    end
end

function [idx1, idx2] = getPlayerSelection(memoramaArray, matchedIndices)
    % Función para obtener la selección de dos índices válidos del jugador
    while true
        try
            selection = input('Seleccione dos números entre 1 y 50 separados por espacio: ', 's');
            nums = sscanf(selection, '%d %d');
            if length(nums) ~=2
                error('Debe seleccionar exactamente dos números.');
            end
            idx1 = nums(1);
            idx2 = nums(2);
            if any(idx1 <1 | idx1 >50 | idx2 <1 | idx2 >50)
                error('Los números deben estar entre 1 y 50.');
            end
            if idx1 == idx2
                error('Debe seleccionar dos números diferentes.');
            end
            if matchedIndices(idx1) || matchedIndices(idx2)
                error('Uno o ambos índices ya han sido emparejados.');
            end
            break; % Selección válida
        catch ME
            fprintf('Entrada inválida: %s. Intente nuevamente.\n', ME.message);
        end
    end
end

function counts = getImageClassCounts(Resultados_Totales, imageName)
    % Función para obtener la cantidad de objetos por clase en una imagen
    subset = Resultados_Totales(strcmp(Resultados_Totales.Nombre_Imagen, imageName), :);
    clases = unique(subset.Etiqueta);
    counts = struct();
    for i = 1:length(clases)
        clase = clases{i};
        counts.(clase) = sum(strcmp(subset.Etiqueta, clase));
    end
end

function displaySelectedImages(folderPath, img1, img2, idx1, idx2)
    % Función para mostrar las imágenes seleccionadas en una figura con subplots
    img1Path = fullfile(folderPath, img1);
    img2Path = fullfile(folderPath, img2);
    
    % Verificar que los archivos existen
    if ~isfile(img1Path) || ~isfile(img2Path)
        warning('Una o ambas imágenes seleccionadas no existen en la ruta especificada.');
        return;
    end
    
    % Leer las imágenes
    img1_display = imread(img1Path);
    img2_display = imread(img2Path);
    
    % Crear una figura para mostrar ambas imágenes
    f_selected = figure('Name', 'Imágenes Seleccionadas', 'NumberTitle', 'off', 'Position', [300 300 1200 600]);
    
    subplot(1,2,1);
    imshow(img1_display);
    title(sprintf('Imagen %d: %s', idx1, img1), 'Interpreter', 'none');
    
    subplot(1,2,2);
    imshow(img2_display);
    title(sprintf('Imagen %d: %s', idx2, img2), 'Interpreter', 'none');
    
    % Pausar para que los jugadores vean las imágenes
    pause(3); % Tiempo en segundos; ajusta según preferencia
    
    % Cerrar la figura después de la pausa
    close(f_selected);
end
