clc % limpia pantalla
clear all % limpia todo
close all % cierra todo
warning off all
disp('Programa para clasificación de un vector basado en la mínima distancia')

% NUEVAS CLASES QUE NO SE INTERSECTAN
c1=[-5 -4 -6 -7; 8 9 7 8];   
c2=[-10 -9 -11 -12; -3 -4 -5 -3];   
c3=[-15 -14 -16 -17; -10 -11 -9 -8]; 
c4=[10 11 10 12; 2 3 4 2];  
c5=[14 15 14 16; 0 1 1 2];  
c6=[18 19 17 20; 5 6 4 6];  

% Graficando las clases
figure
plot(c1(1,:),c1(2,:),'ro','MarkerFaceColor','r','MarkerSize',10)
hold on
grid on
plot(c2(1,:),c2(2,:),'go','MarkerFaceColor','g','MarkerSize',10)
plot(c3(1,:),c3(2,:),'bo','MarkerFaceColor','b','MarkerSize',10)
plot(c4(1,:),c4(2,:),'mo','MarkerFaceColor','m','MarkerSize',10)
plot(c5(1,:),c5(2,:),'co','MarkerFaceColor','c','MarkerSize',10)
plot(c6(1,:),c6(2,:),'yo','MarkerFaceColor','y','MarkerSize',10)
legend('Clase 1','Clase 2','Clase 3','Clase 4','Clase 5','Clase 6')

resp = 1;
hVector =[];

while resp == 1 
    % Pide el vector al usuario
    disp('Ingresa los valores del vector :');
    vx=input('x = ');
    vy=input('y = ');
    vector=[vx;vy];
    
    % Calculando los centroides de las clases
    media1=mean(c1,2);
    media2=mean(c2,2);
    media3=mean(c3,2);
    media4=mean(c4,2);
    media5=mean(c5,2);
    media6=mean(c6,2);

    % Calculando las distancias entre los centroides
    dist_centroides = [
        norm(media1 - media2), norm(media1 - media3), norm(media1 - media4), norm(media1 - media5), norm(media1 - media6), norm(media2 - media3), norm(media2 - media4), norm(media2 - media5), norm(media2 - media6), norm(media3 - media4), norm(media3 - media5), norm(media3 - media6), norm(media4 - media5), norm(media4 - media6), norm(media5 - media6)
    ];

    % Umbral para la distancia máxima
    distancia_maxima = (1/2)*(mean(dist_centroides(:)))
    
    % Calculando las distancias entre el vector y los centroides
    dist1=norm(vector-media1);
    dist2=norm(vector-media2);
    dist3=norm(vector-media3);
    dist4=norm(vector-media4);
    dist5=norm(vector-media5);
    dist6=norm(vector-media6);
    
    % Almacenando todas las distancias en un vector
    dist_total=[dist1 dist2 dist3 dist4 dist5 dist6];
    [minimo, clase]=min(dist_total);
    
    % Verifica si la mínima distancia es menor que la distancia máxima
    if minimo <= distancia_maxima
        fprintf('El vector desconocido pertenece a la clase %d\n', clase);
        fprintf('La mínima distancia es de: %f', minimo);
    else
        disp('El vector desconocido no pertenece a ninguna clase');
        fprintf('La mínima distancia es de: %f', minimo);
    end

    % Elimina el vector anterior del gráfico si existe
    if ~isempty(hVector)
        delete(hVector);
    end
    
    % Graficando el vector
    hVector = plot(vector(1,:),vector(2,:),'*k','MarkerSize',10);
    legend('Clase 1','Clase 2','Clase 3','Clase 4','Clase 5','Clase 6','Vector Desconocido')
    grid on   
    resp = input('\n¿Deseas probar otra vez con otro vector? Si = 1 / No = 0 : ');
    
end
hold off
close all % cierra todo
disp(" Fin del programa... ");
