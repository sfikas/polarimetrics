function res = debruitageJihad(X)
%
% Noise removal tool. Input is (i, j) or (i, j, n-of-channels) image.
% Courtesy of Jihad Zallat of ENSPS.
%
% G.Sfikas 25 Mar 2010
if numel(size(X)) == 3
   res = zeros(size(X));
   for K = 1:size(X, 3)
       res(:, :, K) = debruitageJihad(X(:, :, K));
   end
   return;
end
m = 3;
L = 15;
height =size(X, 1);
width = size(X, 2);

[FImage_SP PDF_SP]=Scatter_plot(X, m, L, .5);
cropI = 1 + floor(m/2);
Image1 = imcrop(X, [cropI cropI (width-m) (height-m)]);
restituee = double(Image1) - FImage_SP;
%%
% Fit restored image back to original image dimensions
res = X;
res(cropI + (0:height-m), cropI + (0:width-m)) = restituee;
return;

function [Image_noise,PDF]=Scatter_plot(Image,m,L,P)
%FONCTION D'APPLICATION DE LA "SCATTER PLOT METHOD"
%[Image_noise PDF]=Scatter_plot(Image,m,L,P)
%Image est l'image � traiter
%m est la taille de la fenetre glissante pour le calcul des
%valeurs moyennes et des �carts-types (e.g 3,5,7)
%L est la taille des LXL blocs dans lesquelles le graphe de la fonction
%"�carttype=f(moyenne)" va etre r�parti (e.g 10)
%P proportion de blocs � s�lectionner pour d�terminer les zones
%homog�nes(e.g. 1/3)
%Image_noise est la r�pr�sentation matricielle du bruit dans les zones
%homog�nes
%PDF vecteur pdf du bruit

%R�cup�ration des dimensions de l'image 
[height width]=size(Image);

% d�finition des "images" moyenne locale (Mean) et ecart-type local (Sigma)
Mean=MyMean(Image,m);
Sigma=MyStd(Image,m);

%suppression des bords
cropI=1+floor(m/2);
Mean = imcrop(Mean,[cropI cropI (width-m) (height-m)]);
Sigma = imcrop(Sigma,[cropI cropI (width-m) (height-m)]);


%R�cup�ration des dimensions de l'image apr�s exclusion des valeurs
%�rron�es du bord
Image1=imcrop(Image,[cropI cropI (width-m) (height-m)]);
[height width]=size(Image1);

%CALCUL DE L'APPARTENANCE AUX DIFFERENTES PARTIES DU GRAPHE DE Sigma=f(Mean)
Mask_Bloc=zeros(height,width,L*L);
Somme_Mask=zeros(height,width);
Sum_Bloc=1:1:(L*L);
Seuil_Mean=0:max(Mean(:))/L:max(Mean(:));
%%BUG for numel(Seuil_Mean) = 0 SFIKAS
Seuil_Sigma=0:max(Sigma(:))/L:max(Sigma(:));
%%ADDED min(L, ..)
for k=1:min(L, numel(Seuil_Mean))
    for l=1:min(L, numel(Seuil_Sigma))
    Mask_Bloc(:,:,(l-1)*L+k)=bitand(uint8(bitand((Mean<Seuil_Mean(k)),(Sigma<Seuil_Sigma(l)))),uint8(bitcmp(Somme_Mask,1)));
    Somme_Mask=bitor(Somme_Mask,Mask_Bloc(:,:,(l-1)*L+k));
    Sum_Bloc((l-1)*L+k)=sum(sum(Mask_Bloc(:,:,(l-1)*L+k)));
    end
end

[Sum_Bloc Zone_homo]=sort(Sum_Bloc);
Zone_homo(1:floor(L*L*(1-P)))=[];
m=((2*m+1)^2/((2*m+1)^2-1))^(1/2);%constante interm�diaire
Image_noise=zeros(height,width);%Image du bruit dans les zones homog�nes
for k=1:max(size(Zone_homo))
    Image_noise=Image_noise+Mask_Bloc(:,:,Zone_homo(k)).*(Image1-Mean)*m;
end


%calcul du pdf
bornemin=floor(min(Image_noise(:)));
bornemax=ceil(max(Image_noise(:)));
PDF=zeros(bornemax-bornemin+11,1);
for i=(bornemin-5):1:(bornemax+5)    
    PDF(i+6-bornemin)=sum(sum((Image_noise>i)&(Image_noise<(i+1))));      
end
PDF=PDF/(max(PDF(:)));

return;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%moyenne mat par des blocs de tailles "taille_bloc"

function resultat=MyMean(Mat,taille_bloc)
    moyenneF = @(x) moyenne(x(:));
    resultat=nlfilter(Mat,[taille_bloc taille_bloc],moyenneF);
    
return;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function resultat=MyStd(Mat,taille_bloc)
    ecarttypeF = @(x) ecarttype(x(:));
    resultat=nlfilter(double(Mat),[taille_bloc taille_bloc],ecarttypeF);

return;

function y=moyenne(x)

     %y=(sum(x(:))/prod(size(x)));
 y=median(x(:));
return;

function y=ecarttype(x)
    %y=std(x(:));
    temp = median(x(:));
    temp = x - temp;
    y = 1.486*median(abs(temp(:)));
return;