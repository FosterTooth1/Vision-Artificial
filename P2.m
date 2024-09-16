clc % limpia pantalla
clear all %limpia todo
close all %cierra todo
warning off all

resp_clase = 1;

while resp_clase == 1
    % Preguntar el número de clases y el número de representantes de cada clase
    num_clases = input('\n Ingresa el número de clases: ');
    num_representantes = input('\n Ingresa el número de representantes por cada clase: ');
    
    % Definimos la paleta de colores
    colores = [
        1.0, 0.0, 0.0;  %Rojo
        0.0, 1.0, 0.0;  %Verde
        0.0, 0.0, 1.0;  %Azul
        1.0, 1.0, 0.0;  %Amarillo
        0.0, 1.0, 1.0;  %Cyan
        1.0, 0.0, 1.0;  %Magnta
        0.5, 0.5, 0.5;  %Gris
        1.0, 0.5, 0.0;  %Naranja
        0.5, 0.0, 0.5;  %Purpura
        0.0, 0.5, 0.5;  %Verde azulado
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
    
    %Umbral para la distancia máxima (mitad del promedio de las distancias entre los centroides)
    distancia_maxima = (1/2) * mean(dist_centroides)
    
    %Generando las clases
    for i=1:num_clases
        Clases(i,:,1) = (randn(1,num_representantes) * dispersion(i,1)) + centroides(i,1);
        Clases(i,:,2) = (randn(1,num_representantes) * dispersion(i,2)) + centroides(i,2);
    end
    
    %Grafica de las clases
    figure
    hold on
    grid on
    
    legend_entries = cell(1, num_clases*2 + 1); % Reserva de el espacio para las leyendas de las clases, centroides y del vector

    for i=1:num_clases
        color_idx = mod(i-1, num_colores) + 1; % Reutiliza los colores ciclicamente
        plot(Clases(i,:,1),Clases(i,:,2),'ko','MarkerFaceColor',colores(color_idx,:),'MarkerSize',10);
        legend_entries{i} = ['Clase ' num2str(i)]; % Guardar la leyenda para cada clase
    end

    for i=1:num_clases
        % Grafica del centroide
        color_idx = mod(i-1, num_colores) + 1; % Reutiliza los colores ciclicamente
        plot(centroides(i,1),centroides(i,2),'ks','MarkerFaceColor',colores(color_idx,:),'MarkerSize',12,'LineWidth',2);
        legend_entries{num_clases + i} = ['Centroide Clase ' num2str(i)]; % Guardar la leyenda para cada centroide
    end
    
    resp_vect = 1;
    hVector =[];
    
    while resp_vect == 1 
        % Pide el vector al usuario
        disp('Ingresa los valores del vector :');
        vx = input('x = ');
        vy = input('y = ');
        vector = [vx, vy];
        
        % Calculando las distancias entre el vector y cada uno de los centroides
        distancias = zeros(num_clases, 1);
        for i = 1:num_clases
            distancias(i) = norm(vector - centroides(i,:));  % Distancia al i-ésimo centroide
        end
        
        %Obteniendo la mínima distancia
        [minimo, clase] = min(distancias);
        
        % Verifica si la mínima distancia es menor que la distancia máxima
        if minimo <= distancia_maxima
            fprintf('\n El vector desconocido pertenece a la clase %d', clase(1));
            fprintf('\n La mínima distancia es de: %f', minimo(1));
        else
            disp('El vector desconocido no pertenece a ninguna clase');
            fprintf('\n La mínima distancia es de: %f', minimo(1));
        end
    
        % Elimina el vector anterior del gráfico si existe
        if ~isempty(hVector)
            delete(hVector);
        end
        
        % Graficando el vector
        hVector = plot(vector(1,1),vector(1,2),'*k','MarkerSize',10);
        legend_entries{num_clases*2 + 1} = 'Vector ingresado'; % Agregar la leyenda para el vector ingresado
        legend(legend_entries); % Crear la leyenda
        grid on   
        resp_vect = input('\n¿Deseas probar otra vez con otro vector? Si = 1 / No = 0 : ');
        
    end
    hold off
    resp_clase = input('\n ¿Deseas probar con otras clases? Si = 1 / No = 0 : ');
    close all

end

disp('ADIOS');
