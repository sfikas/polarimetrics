function [f V z X initialEstimate] = restorePolarimetricImage(IN, H, cheat, K, cheatInit)
% f = restorePolarimetricImage(IN)
%
% Modele a la parametrization "lambda", ou modele "Juillet"
%
% Restore the distorted and noise-corrupted IN, to the original stokes 4-channel image.
% Input argument 1 and output is a X x Y x 4 matrix.
% Second input is the distortion matrix (the H in Hf + n)
%
% Notes
%   Cette version est plutot differente que cette qu'a ete programme en
%   Janvier.
%   Les differences majeurs sont:
%   -Le modele Bayesien maintenant est plus pareil au modele
%   introdui a l'article envoye a CVPR, c'est-a-dire les differences entre
%   les cont.mixing proportions sont modelises comme Student-t.
%   -Le parametrisation des quelques variables aleatoires importantes est
%   bien different. (vois code..)
%   
%   Notez aussi que le variable 'X' qui se trouve dans ce code-ci aparrait
%   comme 'l' dans mes notes et dans l'article correspondant futur.
%
% Update Mardi 19 Mai 09:
%       Allagi ston metasximatismo opws ton zitaei o Christian.
%       Diladi:
%           prosthetw enan oro 1/2 ston yparxon metasxhmatismo
%           kanw permutation tis metavlites s2 <- s3 <- s4 
%
% Update Samedi 14 Fev 09: Will return also the initial estimate
%                           (typically the pseudo-inverse)
%        Mardi 17 Fev 09: Will print ISNR values.
%        Mardi 4 Mar 09: Will actually use the pseudoinverse.. so
%                           now it works for non-invertible H.
% G.Sfikas 6 Juin 2008
%
% JULY 2010 UPDATE: Todo CLEANUP & put in sfikasLibrary
%
% SEP26 2010 UPDATE: Must CLEANUP

flagGridScan = 0;
%%Compute quelques quantites necessaires
NhoodSize = 4;
NhoodDirections = 2;
if exist('K', 'var') == 0
    K = 3;
end
D = 4;
imageSize = size(IN);
if imageSize(3) >= 4
    fprintf('Input is not a valid set of 4-channel polarimetrics image.\n');
    fprintf('Input consists of %d channels. Continuing..\n', imageSize(3));
end
Dinput = imageSize(3);
imageSize(3) = [];
N = prod(imageSize);
g = convertJxN(IN);
cheat = convertJxN(cheat);
Hspflag = 0;
if numel(size(H)) > 2
    disp('Using spatially-variant blur matrix H.');
    Hspflag = 1;
    Hsp = convertJxN(H);
end
if exist('cheatInit', 'var') == 0
    if Hspflag == 0
        f = pinv(H)*g; %this is just an initialization
    else
        for n = 1:N
            f(:, n) = inv(reshape(Hsp(:,n), [4 4])) * g(:, n);
        end
    end
else
    f = cheatInit;
end
initialEstimate = f;
tempE = (f - cheat).*(f - cheat);
E = inv(N*D) * sum(tempE(:));
fprintf('-- Initial error (using H^-1 * g) -- : %f\n', E);
[junk ISNR] = computePolarimetricIndex(f(:), cheat(:), IN(:));
fprintf('-- Initial improvement over SNR (using H^-1 * g) -- : %f\n', ISNR);
fprintf('-- Admissible vectors: %d out of %d.\n', sum(checkStokeValidity(initialEstimate)), N);
%% Upd190509 -- New transform (facteur de 1/2)
f = 2*f;
%%
X(1, :) = sqrt(f(1, :) + f(4, :));
%Il faut que je change la transformation des vecteurs f en X
%Vaut-il mieux que je me serve d'un Cholesky decompo.. voir des notes
l1 = X(1, :);
X(1, l1<0 | imag(l1) ~= 0) = 0;
X(3, :) = f(2, :) ./ (X(1, :) + 1e-3);
X(4, :) = f(3, :) ./ (X(1, :) + 1e-3);
X(2, :) = sqrt(f(1, :) - f(4, :) - X(3, :).^2 - X(4, :).^2);
l2 = X(2, :);
X(2, l2<0 | imag(l2) ~= 0) = 0;
% Initialization
covar = zeros(D, D, K);
allDataCov = cov(X');
% Deterministic initialization -- K-means
[m w] = deterministicKmeans(X(:, 1:N), K);
w = w'*ones(1, N);
%  w = w + 0.2*rand(K, N); % Add some noise..
%  disp('Warning: Added noise on init!');
w = (w ./ (ones(K, 1) * (sum(w, 1)+eps)))';
for i = 1:K
    % Make each std.deviation equal to 1/K of total std.deviation.
    covar(:,:,i) = allDataCov / K^2 + eps*eye(D);
end
beta = double(0.8 * ones(NhoodDirections, K));
u = double(ones(NhoodSize, K, N));
logU = zeros(NhoodSize, K, N);
v = double(ones(NhoodDirections, K));
z = inv(N) * ones(N, K);
wDiff2 = zeros(NhoodSize, K, N);
wDiff2Beta = zeros(NhoodSize, K, N);
likelihood = 0;
V = 1e-6*eye(Dinput);
%
% Main EM loop
%
tic;
fprintf('Restoration using Student-t continuous LP, gmm assumed on lambda vectors (a Stokes parametrization)\n');
fprintf('---------------------------------------------------------------------------------------------------\n');
for iterations = 1:7
    prev = likelihood;
    likelihood = 0;
    % E-step
    for i = 1:K
        z(:, i) = w(:, i) .* gaussianValue(X, m(:, i), covar(:,:,i)) + eps;
        likelihood = likelihood + z(:, i);        
    end

    likelihood = sum(log(likelihood)); %logp(X|Pi), il nous faudra aussi +log(p(Pi))        
    z = z ./ (sum(z, 2) * ones(1, K));
    %% Special -- reinitialize w to the posterior
    if iterations == 1
        im=zeros(imageSize);
        for kk=1:K
            im(:,:) = reshape(z(:, kk), imageSize);
            xx=filter2((1./9.)*ones(3,3),im);
            z(:, kk)=xx(:);
        end
        w = z;
    end
    % M-step
    for i = 1:K
        if(sum(z(:,i)) > 0) %To catch empty clusters
            m(:, i) = (X * z(:, i)) / sum(z(:, i));
        end
        newCovar = inv(sum(z(:, i))) * ( ...
            (X - m(:, i)*ones(1,N))* ...
            sparse(1:N, 1:N, z(:, i), N, N) * ...
            (X - m(:, i)*ones(1,N))' );
        if(isnan(rcond(newCovar))) %for emergencies really!
            newCovar = eye(D);
        end
        newCovar = newCovar + (1e-5)*eye(D);
        if rcond(newCovar) > 1e3 * eps && det(newCovar) > 1e-20
            covar(:, :, i) = newCovar;
        end
        clear newCovar;
    end
    %% Compute q*(U) %%
    % Some convenient statistics, regarding w. 
    %Neighbour square differences: wDiff2
    for j = 1:K
        for n = 1:NhoodSize
            [pos direction] = getNeighbourInfo(n);
            temp = translation(reshape(w(:, j), imageSize),  -pos);
            temp2 = translation(temp, +pos);
            wDiff2(n, j, :) = (temp(:) - temp2(:)).^2;
            wDiff2Beta(n, j, :) = wDiff2(n, j, :) / (beta(direction, j) + eps);
            likelihood = likelihood + ...
                sum(logStudentValue((temp(:) - temp2(:))', 0, beta(direction, j) + eps, v(direction, j)));
        end
    end
    for j = 1:K
        for n = 1:NhoodSize
            u(n, j, :) = (v(j) + 1) ./ (v(j) + wDiff2Beta(n, j, :));
            logU(n, j, :) = ...
                psi((v(j) + 1)/2) - log((v(j) + wDiff2Beta(n, j, :))/2);
        end
    end    
    % STEP 2: Maximize parameters: beta, v (freedom degrees) and pi
    %% Pi (weights) %%
    maxLevel = max([floor(log2(max(imageSize)/16)); 3]);
    if flagGridScan == 0
        oldW = w;    
        for j = 1:K
            aQuad = 0; bQuad = 0;
            cQuad = -0.5 * z(:, j); 
            for k = 1:NhoodSize
                [pos d] = getNeighbourInfo(k);
                temp = translation(reshape(oldW(:, j), imageSize),  -pos);
                aQuad = aQuad + inv(beta(d, j) + eps) * squeeze(u(k, j, :));
                bQuad = bQuad - inv(beta(d, j) + eps) * ...
                    (squeeze(u(k, j, :)) .* temp(:));
            end
            w(:, j) = solveQuad(aQuad, bQuad, cQuad);
        end
        for n = 1:N
            w(n, :) = BIDProjection(w(n, :));
        end
    else
        for mrLevel = maxLevel:-1:0
            w = MEXgridScan(mrLevel, imageSize, w, z, u, beta, K, flagGridScanAlsoZ);
        end
    end
    %% Beta %%
    for j = 1:K
        betaComponent = zeros(1, NhoodSize);
        for n = 1:NhoodSize
            betaComponent(n) = (sum(u(n, j, :) .* wDiff2(n, j, :))) / ...
                    getTotalNeighbours(getNeighbourInfo(n), imageSize);
        end
        for d = 1:size(beta, 1)
            beta(d, j) = sum(betaComponent(getDirectionInfo(d)));
        end
        clear betaComponent;
    end
    %% v (Degrees of freedom) %%
    for j = 1:K
        vComponent = zeros(1, NhoodSize);
        for n = 1:NhoodSize
            vComponent(n) = (sum(logU(n, j, :) - u(n, j, :)) / ...
                getTotalNeighbours(getNeighbourInfo(n), imageSize));
        end
        for d = 1:size(v, 1)
            vConstant = sum(vComponent(getDirectionInfo(d))) + 1;
            v(d, j) = fzero(@(vTemp) vFunction(vTemp, vConstant),[+eps +inv(eps)]);
        end
        clear vComponent;
    end
    %% X re-estimation %%
    % faire attention que pour chaque X doit se tenir que
    % x(1) >= 0, x(2) >= 0
    Om1 = diag(inv(V)); 
    if Hspflag == 0
        h = H;
        %% Upd190509 New transform
        %% H allagh tou "H" einai plasmatiki: ginetai mono gia pragmatopoihthei o
        %% neos metasxhmatismos programmatistika pio efkola.
        h = 2*h;        
    end
    badOptCount = 0; allOptCount = 0; badOpt = 0;
    for n = 1:N
        if Hspflag == 1
            h = reshape(Hsp(:, n), [4 4]);
            %% Upd190509 New transform
            %% H allagh tou "H" einai plasmatiki: ginetai mono gia pragmatopoihthei o
            %% neos metasxhmatismos programmatistika pio efkola.
            h = 2*h;
        end
        %%
        l = X(:, n);
        Om2 = 0;
        Om3 = -2 * g(:, n)' * inv(V);
        Om4 = 0;
        for j = 1:K
            Om2 = Om2 + z(n, j)*inv(covar(:,:,j) + 1e-4);
            Om4 = Om4 - 2*z(n, j)*m(:, j)'*inv(covar(:,:,j) + 1e-4);
        end
        [prevValue prevValueA] = evaluateObjective(g(:, n), h, l, V, m, covar, z(n, :), Om1, Om2, Om3, Om4);         
        for lambdaIter = 1:15
            prevL = l;
            %% Update lambda 1 %%
            %1/4*Om1_1*(h_11 + h_14)^2 + 1/4*Om1_2*(h_21 + h_24)^2 + 1/4*Om1_3*(h_31 + h_34)^2 + 1/4*Om1_4*(h_41 + h_44)^2
            %Om1_1*(h_12*l_3 + h_13*l_4)*(h_11 + h_14) + Om1_2*(h_22*l_3 + h_23*l_4)*(h_21 + h_24) + Om1_3*(h_32*l_3 + h_33*l_4)*(h_31 + h_34) + Om1_4*(h_42*l_3 + h_43*l_4)*(h_41 + h_44)
            %1/2*Om1_1*h_11^2*l_2^2 + 1/2*Om1_1*h_11^2*l_3^2 + 1/2*Om1_1*h_11^2*l_4^2 + 1/2*Om3_1*h_11 + Om1_1*h_12^2*l_3^2 + 2*Om1_1*h_12*h_13*l_3*l_4 + Om1_1*h_13^2*l_4^2 - 1/2*Om1_1*h_14^2*l_2^2 - 1/2*Om1_1*h_14^2*l_3^2 - 1/2*Om1_1*h_14^2*l_4^2 + 1/2*Om3_1*h_14 + 1/2*Om1_2*h_21^2*l_2^2 + 1/2*Om1_2*h_21^2*l_3^2 + 1/2*Om1_2*h_21^2*l_4^2 + 1/2*Om3_2*h_21 + Om1_2*h_22^2*l_3^2 + 2*Om1_2*h_22*h_23*l_3*l_4 + Om1_2*h_23^2*l_4^2 - 1/2*Om1_2*h_24^2*l_2^2 - 1/2*Om1_2*h_24^2*l_3^2 - 1/2*Om1_2*h_24^2*l_4^2 + 1/2*Om3_2*h_24 + 1/2*Om1_3*h_31^2*l_2^2 + 1/2*Om1_3*h_31^2*l_3^2 + 1/2*Om1_3*h_31^2*l_4^2 + 1/2*Om3_3*h_31 + Om1_3*h_32^2*l_3^2 + 2*Om1_3*h_32*h_33*l_3*l_4 + Om1_3*h_33^2*l_4^2 - 1/2*Om1_3*h_34^2*l_2^2 - 1/2*Om1_3*h_34^2*l_3^2 - 1/2*Om1_3*h_34^2*l_4^2 + 1/2*Om3_3*h_34 + 1/2*Om1_4*h_41^2*l_2^2 + 1/2*Om1_4*h_41^2*l_3^2 + 1/2*Om1_4*h_41^2*l_4^2 + 1/2*Om3_4*h_41 + Om1_4*h_42^2*l_3^2 + 2*Om1_4*h_42*h_43*l_3*l_4 + Om1_4*h_43^2*l_4^2 - 1/2*Om1_4*h_44^2*l_2^2 - 1/2*Om1_4*h_44^2*l_3^2 - 1/2*Om1_4*h_44^2*l_4^2 + 1/2*Om3_4*h_44 + Om2_11
            %Om4_1 + 2*Om2_21*l_2 + 2*Om2_31*l_3 + 2*Om2_41*l_4 + Om3_1*(h_12*l_3 + h_13*l_4) + Om3_2*(h_22*l_3 + h_23*l_4) + Om3_3*(h_32*l_3 + h_33*l_4) + Om3_4*(h_42*l_3 + h_43*l_4) + Om1_1*(h_11 - h_14)*(h_12*l_3 + h_13*l_4)*(l_2^2 + l_3^2 + l_4^2) + Om1_2*(h_21 - h_24)*(h_22*l_3 + h_23*l_4)*(l_2^2 + l_3^2 + l_4^2) + Om1_3*(h_31 - h_34)*(h_32*l_3 + h_33*l_4)*(l_2^2 + l_3^2 + l_4^2) + Om1_4*(h_41 - h_44)*(h_42*l_3 + h_43*l_4)*(l_2^2 + l_3^2 + l_4^2)
            coeff(4) = 1/4*Om1(1)*(h(1,1) + h(1,4))^2 + 1/4*Om1(2)*(h(2,1) + h(2,4))^2 + 1/4*Om1(3)*(h(3,1) + h(3,4))^2 + 1/4*Om1(4)*(h(4,1) + h(4,4))^2;
            coeff(3) = Om1(1)*(h(1,2)*l(3) + h(1,3)*l(4))*(h(1,1) + h(1,4)) + Om1(2)*(h(2,2)*l(3) + h(2,3)*l(4))*(h(2,1) + h(2,4)) + Om1(3)*(h(3,2)*l(3) + h(3,3)*l(4))*(h(3,1) + h(3,4)) + Om1(4)*(h(4,2)*l(3) + h(4,3)*l(4))*(h(4,1) + h(4,4));
            coeff(2) = 1/2*Om1(1)*h(1,1)^2*l(2)^2 + 1/2*Om1(1)*h(1,1)^2*l(3)^2 + 1/2*Om1(1)*h(1,1)^2*l(4)^2 + 1/2*Om3(1)*h(1,1) + Om1(1)*h(1,2)^2*l(3)^2 + 2*Om1(1)*h(1,2)*h(1,3)*l(3)*l(4) + Om1(1)*h(1,3)^2*l(4)^2 - 1/2*Om1(1)*h(1,4)^2*l(2)^2 - 1/2*Om1(1)*h(1,4)^2*l(3)^2 - 1/2*Om1(1)*h(1,4)^2*l(4)^2 + 1/2*Om3(1)*h(1,4) + 1/2*Om1(2)*h(2,1)^2*l(2)^2 + 1/2*Om1(2)*h(2,1)^2*l(3)^2 + 1/2*Om1(2)*h(2,1)^2*l(4)^2 + 1/2*Om3(2)*h(2,1) + Om1(2)*h(2,2)^2*l(3)^2 + 2*Om1(2)*h(2,2)*h(2,3)*l(3)*l(4) + Om1(2)*h(2,3)^2*l(4)^2 - 1/2*Om1(2)*h(2,4)^2*l(2)^2 - 1/2*Om1(2)*h(2,4)^2*l(3)^2 - 1/2*Om1(2)*h(2,4)^2*l(4)^2 + 1/2*Om3(2)*h(2,4) + 1/2*Om1(3)*h(3,1)^2*l(2)^2 + 1/2*Om1(3)*h(3,1)^2*l(3)^2 + 1/2*Om1(3)*h(3,1)^2*l(4)^2 + 1/2*Om3(3)*h(3,1) + Om1(3)*h(3,2)^2*l(3)^2 + 2*Om1(3)*h(3,2)*h(3,3)*l(3)*l(4) + Om1(3)*h(3,3)^2*l(4)^2 - 1/2*Om1(3)*h(3,4)^2*l(2)^2 - 1/2*Om1(3)*h(3,4)^2*l(3)^2 - 1/2*Om1(3)*h(3,4)^2*l(4)^2 + 1/2*Om3(3)*h(3,4) + 1/2*Om1(4)*h(4,1)^2*l(2)^2 + 1/2*Om1(4)*h(4,1)^2*l(3)^2 + 1/2*Om1(4)*h(4,1)^2*l(4)^2 + 1/2*Om3(4)*h(4,1) + Om1(4)*h(4,2)^2*l(3)^2 + 2*Om1(4)*h(4,2)*h(4,3)*l(3)*l(4) + Om1(4)*h(4,3)^2*l(4)^2 - 1/2*Om1(4)*h(4,4)^2*l(2)^2 - 1/2*Om1(4)*h(4,4)^2*l(3)^2 - 1/2*Om1(4)*h(4,4)^2*l(4)^2 + 1/2*Om3(4)*h(4,4) + Om2(1,1);
            coeff(1) = Om4(1) + 2*Om2(2,1)*l(2) + 2*Om2(3,1)*l(3) + 2*Om2(4,1)*l(4) + Om3(1)*(h(1,2)*l(3) + h(1,3)*l(4)) + Om3(2)*(h(2,2)*l(3) + h(2,3)*l(4)) + Om3(3)*(h(3,2)*l(3) + h(3,3)*l(4)) + Om3(4)*(h(4,2)*l(3) + h(4,3)*l(4)) + Om1(1)*(h(1,1) - h(1,4))*(h(1,2)*l(3) + h(1,3)*l(4))*(l(2)^2 + l(3)^2 + l(4)^2) + Om1(2)*(h(2,1) - h(2,4))*(h(2,2)*l(3) + h(2,3)*l(4))*(l(2)^2 + l(3)^2 + l(4)^2) + Om1(3)*(h(3,1) - h(3,4))*(h(3,2)*l(3) + h(3,3)*l(4))*(l(2)^2 + l(3)^2 + l(4)^2) + Om1(4)*(h(4,1) - h(4,4))*(h(4,2)*l(3) + h(4,3)*l(4))*(l(2)^2 + l(3)^2 + l(4)^2);
            theRoots = roots([4*coeff(4) 3*coeff(3) 2*coeff(2) coeff(1)]);
            theRoots = theRoots(imag(theRoots) == 0);
            theRoots(theRoots < 0) = 0;
            oldL = l(1);
            l(1) = bestRoot(theRoots, coeff);
            [nextValue nextValueA] = evaluateObjective(g(:, n), h, l, V, m, covar, z(n, :), Om1, Om2, Om3, Om4);
%             if prevValue < nextValue
%                 badOpt = badOpt + nextValue - prevValue;                
%                 l(1) = oldL; badOptCount = badOptCount + 1;
%             end            
            prevValue = nextValue; prevValueA = nextValueA;           
            %% Update lambda 2 %%
            %1/4*Om1_1*(h_11 - h_14)^2 + 1/4*Om1_2*(h_21 - h_24)^2 + 1/4*Om1_3*(h_31 - h_34)^2 + 1/4*Om1_4*(h_41 - h_44)^2
            %0
            %1/2*Om1_1*h_11^2*l_1^2 + 1/2*Om1_1*h_11^2*l_3^2 + 1/2*Om1_1*h_11^2*l_4^2 - Om1_1*h_11*h_14*l_3^2 - Om1_1*h_11*h_14*l_4^2 + Om1_1*h_12*h_11*l_1*l_3 + Om1_1*h_13*h_11*l_1*l_4 + 1/2*Om3_1*h_11 - 1/2*Om1_1*h_14^2*l_1^2 + 1/2*Om1_1*h_14^2*l_3^2 + 1/2*Om1_1*h_14^2*l_4^2 - Om1_1*h_12*h_14*l_1*l_3 - Om1_1*h_13*h_14*l_1*l_4 - 1/2*Om3_1*h_14 + 1/2*Om1_2*h_21^2*l_1^2 + 1/2*Om1_2*h_21^2*l_3^2 + 1/2*Om1_2*h_21^2*l_4^2 - Om1_2*h_21*h_24*l_3^2 - Om1_2*h_21*h_24*l_4^2 + Om1_2*h_22*h_21*l_1*l_3 + Om1_2*h_23*h_21*l_1*l_4 + 1/2*Om3_2*h_21 - 1/2*Om1_2*h_24^2*l_1^2 + 1/2*Om1_2*h_24^2*l_3^2 + 1/2*Om1_2*h_24^2*l_4^2 - Om1_2*h_22*h_24*l_1*l_3 - Om1_2*h_23*h_24*l_1*l_4 - 1/2*Om3_2*h_24 + 1/2*Om1_3*h_31^2*l_1^2 + 1/2*Om1_3*h_31^2*l_3^2 + 1/2*Om1_3*h_31^2*l_4^2 - Om1_3*h_31*h_34*l_3^2 - Om1_3*h_31*h_34*l_4^2 + Om1_3*h_32*h_31*l_1*l_3 + Om1_3*h_33*h_31*l_1*l_4 + 1/2*Om3_3*h_31 - 1/2*Om1_3*h_34^2*l_1^2 + 1/2*Om1_3*h_34^2*l_3^2 + 1/2*Om1_3*h_34^2*l_4^2 - Om1_3*h_32*h_34*l_1*l_3 - Om1_3*h_33*h_34*l_1*l_4 - 1/2*Om3_3*h_34 + 1/2*Om1_4*h_41^2*l_1^2 + 1/2*Om1_4*h_41^2*l_3^2 + 1/2*Om1_4*h_41^2*l_4^2 - Om1_4*h_41*h_44*l_3^2 - Om1_4*h_41*h_44*l_4^2 + Om1_4*h_42*h_41*l_1*l_3 + Om1_4*h_43*h_41*l_1*l_4 + 1/2*Om3_4*h_41 - 1/2*Om1_4*h_44^2*l_1^2 + 1/2*Om1_4*h_44^2*l_3^2 + 1/2*Om1_4*h_44^2*l_4^2 - Om1_4*h_42*h_44*l_1*l_3 - Om1_4*h_43*h_44*l_1*l_4 - 1/2*Om3_4*h_44 + Om2_22
            %Om4_2 + 2*Om2_21*l_1 + 2*Om2_32*l_3 + 2*Om2_42*l_4
            coeff(4) = 1/4*Om1(1)*(h(1,1) - h(1,4))^2 + 1/4*Om1(2)*(h(2,1) - h(2,4))^2 + 1/4*Om1(3)*(h(3,1) - h(3,4))^2 + 1/4*Om1(4)*(h(4,1) - h(4,4))^2;
            coeff(3) = 0;
            coeff(2) = 1/2*Om1(1)*h(1,1)^2*l(1)^2 + 1/2*Om1(1)*h(1,1)^2*l(3)^2 + 1/2*Om1(1)*h(1,1)^2*l(4)^2 - Om1(1)*h(1,1)*h(1,4)*l(3)^2 - Om1(1)*h(1,1)*h(1,4)*l(4)^2 + Om1(1)*h(1,2)*h(1,1)*l(1)*l(3) + Om1(1)*h(1,3)*h(1,1)*l(1)*l(4) + 1/2*Om3(1)*h(1,1) - 1/2*Om1(1)*h(1,4)^2*l(1)^2 + 1/2*Om1(1)*h(1,4)^2*l(3)^2 + 1/2*Om1(1)*h(1,4)^2*l(4)^2 - Om1(1)*h(1,2)*h(1,4)*l(1)*l(3) - Om1(1)*h(1,3)*h(1,4)*l(1)*l(4) - 1/2*Om3(1)*h(1,4) + 1/2*Om1(2)*h(2,1)^2*l(1)^2 + 1/2*Om1(2)*h(2,1)^2*l(3)^2 + 1/2*Om1(2)*h(2,1)^2*l(4)^2 - Om1(2)*h(2,1)*h(2,4)*l(3)^2 - Om1(2)*h(2,1)*h(2,4)*l(4)^2 + Om1(2)*h(2,2)*h(2,1)*l(1)*l(3) + Om1(2)*h(2,3)*h(2,1)*l(1)*l(4) + 1/2*Om3(2)*h(2,1) - 1/2*Om1(2)*h(2,4)^2*l(1)^2 + 1/2*Om1(2)*h(2,4)^2*l(3)^2 + 1/2*Om1(2)*h(2,4)^2*l(4)^2 - Om1(2)*h(2,2)*h(2,4)*l(1)*l(3) - Om1(2)*h(2,3)*h(2,4)*l(1)*l(4) - 1/2*Om3(2)*h(2,4) + 1/2*Om1(3)*h(3,1)^2*l(1)^2 + 1/2*Om1(3)*h(3,1)^2*l(3)^2 + 1/2*Om1(3)*h(3,1)^2*l(4)^2 - Om1(3)*h(3,1)*h(3,4)*l(3)^2 - Om1(3)*h(3,1)*h(3,4)*l(4)^2 + Om1(3)*h(3,2)*h(3,1)*l(1)*l(3) + Om1(3)*h(3,3)*h(3,1)*l(1)*l(4) + 1/2*Om3(3)*h(3,1) - 1/2*Om1(3)*h(3,4)^2*l(1)^2 + 1/2*Om1(3)*h(3,4)^2*l(3)^2 + 1/2*Om1(3)*h(3,4)^2*l(4)^2 - Om1(3)*h(3,2)*h(3,4)*l(1)*l(3) - Om1(3)*h(3,3)*h(3,4)*l(1)*l(4) - 1/2*Om3(3)*h(3,4) + 1/2*Om1(4)*h(4,1)^2*l(1)^2 + 1/2*Om1(4)*h(4,1)^2*l(3)^2 + 1/2*Om1(4)*h(4,1)^2*l(4)^2 - Om1(4)*h(4,1)*h(4,4)*l(3)^2 - Om1(4)*h(4,1)*h(4,4)*l(4)^2 + Om1(4)*h(4,2)*h(4,1)*l(1)*l(3) + Om1(4)*h(4,3)*h(4,1)*l(1)*l(4) + 1/2*Om3(4)*h(4,1) - 1/2*Om1(4)*h(4,4)^2*l(1)^2 + 1/2*Om1(4)*h(4,4)^2*l(3)^2 + 1/2*Om1(4)*h(4,4)^2*l(4)^2 - Om1(4)*h(4,2)*h(4,4)*l(1)*l(3) - Om1(4)*h(4,3)*h(4,4)*l(1)*l(4) - 1/2*Om3(4)*h(4,4) + Om2(2,2);
            coeff(1) = Om4(2) + 2*Om2(2,1)*l(1) + 2*Om2(3,2)*l(3) + 2*Om2(4,2)*l(4);
            theRoots = roots([4*coeff(4) 3*coeff(3) 2*coeff(2) coeff(1)]);
            theRoots = theRoots(imag(theRoots) == 0);
            theRoots(theRoots < 0) = 0;            
            oldL = l(2);    
            l(2) = bestRoot(theRoots, coeff);
            [nextValue nextValueA] = evaluateObjective(g(:, n), h, l, V, m, covar, z(n, :), Om1, Om2, Om3, Om4);
%             if prevValue < nextValue
%                 badOpt = badOpt + nextValue - prevValue;                
%                 l(2) = oldL; badOptCount = badOptCount + 1;
%             end            
            prevValue = nextValue; prevValueA = nextValueA;                       
            %% Update lambda 3 %%
            %1/4*Om1_1*(h_11 - h_14)^2 + 1/4*Om1_2*(h_21 - h_24)^2 + 1/4*Om1_3*(h_31 - h_34)^2 + 1/4*Om1_4*(h_41 - h_44)^2
            %l_1*(Om1_1*h_11*h_12 - Om1_1*h_12*h_14 + Om1_2*h_21*h_22 - Om1_2*h_22*h_24 + Om1_3*h_31*h_32 - Om1_3*h_32*h_34 + Om1_4*h_41*h_42 - Om1_4*h_42*h_44)
            %1/2*Om1_1*h_11^2*l_1^2 + 1/2*Om1_1*h_11^2*l_2^2 + 1/2*Om1_1*h_11^2*l_4^2 - Om1_1*h_11*h_14*l_2^2 - Om1_1*h_11*h_14*l_4^2 + Om1_1*h_13*h_11*l_1*l_4 + 1/2*Om3_1*h_11 + Om1_1*h_12^2*l_1^2 - 1/2*Om1_1*h_14^2*l_1^2 + 1/2*Om1_1*h_14^2*l_2^2 + 1/2*Om1_1*h_14^2*l_4^2 - Om1_1*h_13*h_14*l_1*l_4 - 1/2*Om3_1*h_14 + 1/2*Om1_2*h_21^2*l_1^2 + 1/2*Om1_2*h_21^2*l_2^2 + 1/2*Om1_2*h_21^2*l_4^2 - Om1_2*h_21*h_24*l_2^2 - Om1_2*h_21*h_24*l_4^2 + Om1_2*h_23*h_21*l_1*l_4 + 1/2*Om3_2*h_21 + Om1_2*h_22^2*l_1^2 - 1/2*Om1_2*h_24^2*l_1^2 + 1/2*Om1_2*h_24^2*l_2^2 + 1/2*Om1_2*h_24^2*l_4^2 - Om1_2*h_23*h_24*l_1*l_4 - 1/2*Om3_2*h_24 + 1/2*Om1_3*h_31^2*l_1^2 + 1/2*Om1_3*h_31^2*l_2^2 + 1/2*Om1_3*h_31^2*l_4^2 - Om1_3*h_31*h_34*l_2^2 - Om1_3*h_31*h_34*l_4^2 + Om1_3*h_33*h_31*l_1*l_4 + 1/2*Om3_3*h_31 + Om1_3*h_32^2*l_1^2 - 1/2*Om1_3*h_34^2*l_1^2 + 1/2*Om1_3*h_34^2*l_2^2 + 1/2*Om1_3*h_34^2*l_4^2 - Om1_3*h_33*h_34*l_1*l_4 - 1/2*Om3_3*h_34 + 1/2*Om1_4*h_41^2*l_1^2 + 1/2*Om1_4*h_41^2*l_2^2 + 1/2*Om1_4*h_41^2*l_4^2 - Om1_4*h_41*h_44*l_2^2 - Om1_4*h_41*h_44*l_4^2 + Om1_4*h_43*h_41*l_1*l_4 + 1/2*Om3_4*h_41 + Om1_4*h_42^2*l_1^2 - 1/2*Om1_4*h_44^2*l_1^2 + 1/2*Om1_4*h_44^2*l_2^2 + 1/2*Om1_4*h_44^2*l_4^2 - Om1_4*h_43*h_44*l_1*l_4 - 1/2*Om3_4*h_44 + Om2_33
            %Om4_3 + 2*Om2_31*l_1 + 2*Om2_32*l_2 + 2*Om2_43*l_4 + Om3_1*h_12*l_1 + Om3_2*h_22*l_1 + Om3_3*h_32*l_1 + Om3_4*h_42*l_1 + Om1_1*h_11*h_12*l_1^3 + Om1_1*h_12*h_14*l_1^3 + Om1_2*h_21*h_22*l_1^3 + Om1_2*h_22*h_24*l_1^3 + Om1_3*h_31*h_32*l_1^3 + Om1_3*h_32*h_34*l_1^3 + Om1_4*h_41*h_42*l_1^3 + Om1_4*h_42*h_44*l_1^3 + Om1_1*h_11*h_12*l_1*l_2^2 - Om1_1*h_12*h_14*l_1*l_2^2 + Om1_1*h_11*h_12*l_1*l_4^2 + Om1_2*h_21*h_22*l_1*l_2^2 + 2*Om1_1*h_12*h_13*l_1^2*l_4 - Om1_1*h_12*h_14*l_1*l_4^2 - Om1_2*h_22*h_24*l_1*l_2^2 + Om1_2*h_21*h_22*l_1*l_4^2 + Om1_3*h_31*h_32*l_1*l_2^2 + 2*Om1_2*h_22*h_23*l_1^2*l_4 - Om1_2*h_22*h_24*l_1*l_4^2 - Om1_3*h_32*h_34*l_1*l_2^2 + Om1_3*h_31*h_32*l_1*l_4^2 + Om1_4*h_41*h_42*l_1*l_2^2 + 2*Om1_3*h_32*h_33*l_1^2*l_4 - Om1_3*h_32*h_34*l_1*l_4^2 - Om1_4*h_42*h_44*l_1*l_2^2 + Om1_4*h_41*h_42*l_1*l_4^2 + 2*Om1_4*h_42*h_43*l_1^2*l_4 - Om1_4*h_42*h_44*l_1*l_4^2
            coeff(4) = 1/4*Om1(1)*(h(1,1) - h(1,4))^2 + 1/4*Om1(2)*(h(2,1) - h(2,4))^2 + 1/4*Om1(3)*(h(3,1) - h(3,4))^2 + 1/4*Om1(4)*(h(4,1) - h(4,4))^2;
            coeff(3) = l(1)*(Om1(1)*h(1,1)*h(1,2) - Om1(1)*h(1,2)*h(1,4) + Om1(2)*h(2,1)*h(2,2) - Om1(2)*h(2,2)*h(2,4) + Om1(3)*h(3,1)*h(3,2) - Om1(3)*h(3,2)*h(3,4) + Om1(4)*h(4,1)*h(4,2) - Om1(4)*h(4,2)*h(4,4));
            coeff(2) = 1/2*Om1(1)*h(1,1)^2*l(1)^2 + 1/2*Om1(1)*h(1,1)^2*l(2)^2 + 1/2*Om1(1)*h(1,1)^2*l(4)^2 - Om1(1)*h(1,1)*h(1,4)*l(2)^2 - Om1(1)*h(1,1)*h(1,4)*l(4)^2 + Om1(1)*h(1,3)*h(1,1)*l(1)*l(4) + 1/2*Om3(1)*h(1,1) + Om1(1)*h(1,2)^2*l(1)^2 - 1/2*Om1(1)*h(1,4)^2*l(1)^2 + 1/2*Om1(1)*h(1,4)^2*l(2)^2 + 1/2*Om1(1)*h(1,4)^2*l(4)^2 - Om1(1)*h(1,3)*h(1,4)*l(1)*l(4) - 1/2*Om3(1)*h(1,4) + 1/2*Om1(2)*h(2,1)^2*l(1)^2 + 1/2*Om1(2)*h(2,1)^2*l(2)^2 + 1/2*Om1(2)*h(2,1)^2*l(4)^2 - Om1(2)*h(2,1)*h(2,4)*l(2)^2 - Om1(2)*h(2,1)*h(2,4)*l(4)^2 + Om1(2)*h(2,3)*h(2,1)*l(1)*l(4) + 1/2*Om3(2)*h(2,1) + Om1(2)*h(2,2)^2*l(1)^2 - 1/2*Om1(2)*h(2,4)^2*l(1)^2 + 1/2*Om1(2)*h(2,4)^2*l(2)^2 + 1/2*Om1(2)*h(2,4)^2*l(4)^2 - Om1(2)*h(2,3)*h(2,4)*l(1)*l(4) - 1/2*Om3(2)*h(2,4) + 1/2*Om1(3)*h(3,1)^2*l(1)^2 + 1/2*Om1(3)*h(3,1)^2*l(2)^2 + 1/2*Om1(3)*h(3,1)^2*l(4)^2 - Om1(3)*h(3,1)*h(3,4)*l(2)^2 - Om1(3)*h(3,1)*h(3,4)*l(4)^2 + Om1(3)*h(3,3)*h(3,1)*l(1)*l(4) + 1/2*Om3(3)*h(3,1) + Om1(3)*h(3,2)^2*l(1)^2 - 1/2*Om1(3)*h(3,4)^2*l(1)^2 + 1/2*Om1(3)*h(3,4)^2*l(2)^2 + 1/2*Om1(3)*h(3,4)^2*l(4)^2 - Om1(3)*h(3,3)*h(3,4)*l(1)*l(4) - 1/2*Om3(3)*h(3,4) + 1/2*Om1(4)*h(4,1)^2*l(1)^2 + 1/2*Om1(4)*h(4,1)^2*l(2)^2 + 1/2*Om1(4)*h(4,1)^2*l(4)^2 - Om1(4)*h(4,1)*h(4,4)*l(2)^2 - Om1(4)*h(4,1)*h(4,4)*l(4)^2 + Om1(4)*h(4,3)*h(4,1)*l(1)*l(4) + 1/2*Om3(4)*h(4,1) + Om1(4)*h(4,2)^2*l(1)^2 - 1/2*Om1(4)*h(4,4)^2*l(1)^2 + 1/2*Om1(4)*h(4,4)^2*l(2)^2 + 1/2*Om1(4)*h(4,4)^2*l(4)^2 - Om1(4)*h(4,3)*h(4,4)*l(1)*l(4) - 1/2*Om3(4)*h(4,4) + Om2(3,3);
            coeff(1) = Om4(3) + 2*Om2(3,1)*l(1) + 2*Om2(3,2)*l(2) + 2*Om2(4,3)*l(4) + Om3(1)*h(1,2)*l(1) + Om3(2)*h(2,2)*l(1) + Om3(3)*h(3,2)*l(1) + Om3(4)*h(4,2)*l(1) + Om1(1)*h(1,1)*h(1,2)*l(1)^3 + Om1(1)*h(1,2)*h(1,4)*l(1)^3 + Om1(2)*h(2,1)*h(2,2)*l(1)^3 + Om1(2)*h(2,2)*h(2,4)*l(1)^3 + Om1(3)*h(3,1)*h(3,2)*l(1)^3 + Om1(3)*h(3,2)*h(3,4)*l(1)^3 + Om1(4)*h(4,1)*h(4,2)*l(1)^3 + Om1(4)*h(4,2)*h(4,4)*l(1)^3 + Om1(1)*h(1,1)*h(1,2)*l(1)*l(2)^2 - Om1(1)*h(1,2)*h(1,4)*l(1)*l(2)^2 + Om1(1)*h(1,1)*h(1,2)*l(1)*l(4)^2 + Om1(2)*h(2,1)*h(2,2)*l(1)*l(2)^2 + 2*Om1(1)*h(1,2)*h(1,3)*l(1)^2*l(4) - Om1(1)*h(1,2)*h(1,4)*l(1)*l(4)^2 - Om1(2)*h(2,2)*h(2,4)*l(1)*l(2)^2 + Om1(2)*h(2,1)*h(2,2)*l(1)*l(4)^2 + Om1(3)*h(3,1)*h(3,2)*l(1)*l(2)^2 + 2*Om1(2)*h(2,2)*h(2,3)*l(1)^2*l(4) - Om1(2)*h(2,2)*h(2,4)*l(1)*l(4)^2 - Om1(3)*h(3,2)*h(3,4)*l(1)*l(2)^2 + Om1(3)*h(3,1)*h(3,2)*l(1)*l(4)^2 + Om1(4)*h(4,1)*h(4,2)*l(1)*l(2)^2 + 2*Om1(3)*h(3,2)*h(3,3)*l(1)^2*l(4) - Om1(3)*h(3,2)*h(3,4)*l(1)*l(4)^2 - Om1(4)*h(4,2)*h(4,4)*l(1)*l(2)^2 + Om1(4)*h(4,1)*h(4,2)*l(1)*l(4)^2 + 2*Om1(4)*h(4,2)*h(4,3)*l(1)^2*l(4) - Om1(4)*h(4,2)*h(4,4)*l(1)*l(4)^2;
            theRoots = roots([4*coeff(4) 3*coeff(3) 2*coeff(2) coeff(1)]);
            theRoots = theRoots(imag(theRoots) == 0);
            oldL = l(3);            
            l(3) = bestRoot(theRoots, coeff);
            [nextValue nextValueA] = evaluateObjective(g(:, n), h, l, V, m, covar, z(n, :), Om1, Om2, Om3, Om4);
%             if prevValue < nextValue
%                 badOpt = badOpt + nextValue - prevValue;                
%                 l(3) = oldL; badOptCount = badOptCount + 1;
%             end            
            prevValue = nextValue; prevValueA = nextValueA;                                   
            %% Update lambda 4 %%
            %1/4*Om1_1*(h_11 - h_14)^2 + 1/4*Om1_2*(h_21 - h_24)^2 + 1/4*Om1_3*(h_31 - h_34)^2 + 1/4*Om1_4*(h_41 - h_44)^2
            %l_1*(Om1_1*h_11*h_13 - Om1_1*h_13*h_14 + Om1_2*h_21*h_23 - Om1_2*h_23*h_24 + Om1_3*h_31*h_33 - Om1_3*h_33*h_34 + Om1_4*h_41*h_43 - Om1_4*h_43*h_44)
            %1/2*Om1_1*h_11^2*l_1^2 + 1/2*Om1_1*h_11^2*l_2^2 + 1/2*Om1_1*h_11^2*l_3^2 - Om1_1*h_11*h_14*l_2^2 - Om1_1*h_11*h_14*l_3^2 + Om1_1*h_12*h_11*l_1*l_3 + 1/2*Om3_1*h_11 + Om1_1*h_13^2*l_1^2 - 1/2*Om1_1*h_14^2*l_1^2 + 1/2*Om1_1*h_14^2*l_2^2 + 1/2*Om1_1*h_14^2*l_3^2 - Om1_1*h_12*h_14*l_1*l_3 - 1/2*Om3_1*h_14 + 1/2*Om1_2*h_21^2*l_1^2 + 1/2*Om1_2*h_21^2*l_2^2 + 1/2*Om1_2*h_21^2*l_3^2 - Om1_2*h_21*h_24*l_2^2 - Om1_2*h_21*h_24*l_3^2 + Om1_2*h_22*h_21*l_1*l_3 + 1/2*Om3_2*h_21 + Om1_2*h_23^2*l_1^2 - 1/2*Om1_2*h_24^2*l_1^2 + 1/2*Om1_2*h_24^2*l_2^2 + 1/2*Om1_2*h_24^2*l_3^2 - Om1_2*h_22*h_24*l_1*l_3 - 1/2*Om3_2*h_24 + 1/2*Om1_3*h_31^2*l_1^2 + 1/2*Om1_3*h_31^2*l_2^2 + 1/2*Om1_3*h_31^2*l_3^2 - Om1_3*h_31*h_34*l_2^2 - Om1_3*h_31*h_34*l_3^2 + Om1_3*h_32*h_31*l_1*l_3 + 1/2*Om3_3*h_31 + Om1_3*h_33^2*l_1^2 - 1/2*Om1_3*h_34^2*l_1^2 + 1/2*Om1_3*h_34^2*l_2^2 + 1/2*Om1_3*h_34^2*l_3^2 - Om1_3*h_32*h_34*l_1*l_3 - 1/2*Om3_3*h_34 + 1/2*Om1_4*h_41^2*l_1^2 + 1/2*Om1_4*h_41^2*l_2^2 + 1/2*Om1_4*h_41^2*l_3^2 - Om1_4*h_41*h_44*l_2^2 - Om1_4*h_41*h_44*l_3^2 + Om1_4*h_42*h_41*l_1*l_3 + 1/2*Om3_4*h_41 + Om1_4*h_43^2*l_1^2 - 1/2*Om1_4*h_44^2*l_1^2 + 1/2*Om1_4*h_44^2*l_2^2 + 1/2*Om1_4*h_44^2*l_3^2 - Om1_4*h_42*h_44*l_1*l_3 - 1/2*Om3_4*h_44 + Om2_44
            %Om4_4 + 2*Om2_41*l_1 + 2*Om2_42*l_2 + 2*Om2_43*l_3 + Om3_1*h_13*l_1 + Om3_2*h_23*l_1 + Om3_3*h_33*l_1 + Om3_4*h_43*l_1 + Om1_1*h_11*h_13*l_1^3 + Om1_1*h_13*h_14*l_1^3 + Om1_2*h_21*h_23*l_1^3 + Om1_2*h_23*h_24*l_1^3 + Om1_3*h_31*h_33*l_1^3 + Om1_3*h_33*h_34*l_1^3 + Om1_4*h_41*h_43*l_1^3 + Om1_4*h_43*h_44*l_1^3 + Om1_1*h_11*h_13*l_1*l_2^2 - Om1_1*h_13*h_14*l_1*l_2^2 + Om1_1*h_11*h_13*l_1*l_3^2 + 2*Om1_1*h_12*h_13*l_1^2*l_3 - Om1_1*h_13*h_14*l_1*l_3^2 + Om1_2*h_21*h_23*l_1*l_2^2 - Om1_2*h_23*h_24*l_1*l_2^2 + Om1_2*h_21*h_23*l_1*l_3^2 + 2*Om1_2*h_22*h_23*l_1^2*l_3 - Om1_2*h_23*h_24*l_1*l_3^2 + Om1_3*h_31*h_33*l_1*l_2^2 - Om1_3*h_33*h_34*l_1*l_2^2 + Om1_3*h_31*h_33*l_1*l_3^2 + 2*Om1_3*h_32*h_33*l_1^2*l_3 - Om1_3*h_33*h_34*l_1*l_3^2 + Om1_4*h_41*h_43*l_1*l_2^2 - Om1_4*h_43*h_44*l_1*l_2^2 + Om1_4*h_41*h_43*l_1*l_3^2 + 2*Om1_4*h_42*h_43*l_1^2*l_3 - Om1_4*h_43*h_44*l_1*l_3^2
            coeff(4) = 1/4*Om1(1)*(h(1,1) - h(1,4))^2 + 1/4*Om1(2)*(h(2,1) - h(2,4))^2 + 1/4*Om1(3)*(h(3,1) - h(3,4))^2 + 1/4*Om1(4)*(h(4,1) - h(4,4))^2;
            coeff(3) = l(1)*(Om1(1)*h(1,1)*h(1,3) - Om1(1)*h(1,3)*h(1,4) + Om1(2)*h(2,1)*h(2,3) - Om1(2)*h(2,3)*h(2,4) + Om1(3)*h(3,1)*h(3,3) - Om1(3)*h(3,3)*h(3,4) + Om1(4)*h(4,1)*h(4,3) - Om1(4)*h(4,3)*h(4,4));
            coeff(2) = 1/2*Om1(1)*h(1,1)^2*l(1)^2 + 1/2*Om1(1)*h(1,1)^2*l(2)^2 + 1/2*Om1(1)*h(1,1)^2*l(3)^2 - Om1(1)*h(1,1)*h(1,4)*l(2)^2 - Om1(1)*h(1,1)*h(1,4)*l(3)^2 + Om1(1)*h(1,2)*h(1,1)*l(1)*l(3) + 1/2*Om3(1)*h(1,1) + Om1(1)*h(1,3)^2*l(1)^2 - 1/2*Om1(1)*h(1,4)^2*l(1)^2 + 1/2*Om1(1)*h(1,4)^2*l(2)^2 + 1/2*Om1(1)*h(1,4)^2*l(3)^2 - Om1(1)*h(1,2)*h(1,4)*l(1)*l(3) - 1/2*Om3(1)*h(1,4) + 1/2*Om1(2)*h(2,1)^2*l(1)^2 + 1/2*Om1(2)*h(2,1)^2*l(2)^2 + 1/2*Om1(2)*h(2,1)^2*l(3)^2 - Om1(2)*h(2,1)*h(2,4)*l(2)^2 - Om1(2)*h(2,1)*h(2,4)*l(3)^2 + Om1(2)*h(2,2)*h(2,1)*l(1)*l(3) + 1/2*Om3(2)*h(2,1) + Om1(2)*h(2,3)^2*l(1)^2 - 1/2*Om1(2)*h(2,4)^2*l(1)^2 + 1/2*Om1(2)*h(2,4)^2*l(2)^2 + 1/2*Om1(2)*h(2,4)^2*l(3)^2 - Om1(2)*h(2,2)*h(2,4)*l(1)*l(3) - 1/2*Om3(2)*h(2,4) + 1/2*Om1(3)*h(3,1)^2*l(1)^2 + 1/2*Om1(3)*h(3,1)^2*l(2)^2 + 1/2*Om1(3)*h(3,1)^2*l(3)^2 - Om1(3)*h(3,1)*h(3,4)*l(2)^2 - Om1(3)*h(3,1)*h(3,4)*l(3)^2 + Om1(3)*h(3,2)*h(3,1)*l(1)*l(3) + 1/2*Om3(3)*h(3,1) + Om1(3)*h(3,3)^2*l(1)^2 - 1/2*Om1(3)*h(3,4)^2*l(1)^2 + 1/2*Om1(3)*h(3,4)^2*l(2)^2 + 1/2*Om1(3)*h(3,4)^2*l(3)^2 - Om1(3)*h(3,2)*h(3,4)*l(1)*l(3) - 1/2*Om3(3)*h(3,4) + 1/2*Om1(4)*h(4,1)^2*l(1)^2 + 1/2*Om1(4)*h(4,1)^2*l(2)^2 + 1/2*Om1(4)*h(4,1)^2*l(3)^2 - Om1(4)*h(4,1)*h(4,4)*l(2)^2 - Om1(4)*h(4,1)*h(4,4)*l(3)^2 + Om1(4)*h(4,2)*h(4,1)*l(1)*l(3) + 1/2*Om3(4)*h(4,1) + Om1(4)*h(4,3)^2*l(1)^2 - 1/2*Om1(4)*h(4,4)^2*l(1)^2 + 1/2*Om1(4)*h(4,4)^2*l(2)^2 + 1/2*Om1(4)*h(4,4)^2*l(3)^2 - Om1(4)*h(4,2)*h(4,4)*l(1)*l(3) - 1/2*Om3(4)*h(4,4) + Om2(4,4);
            coeff(1) = Om4(4) + 2*Om2(4,1)*l(1) + 2*Om2(4,2)*l(2) + 2*Om2(4,3)*l(3) + Om3(1)*h(1,3)*l(1) + Om3(2)*h(2,3)*l(1) + Om3(3)*h(3,3)*l(1) + Om3(4)*h(4,3)*l(1) + Om1(1)*h(1,1)*h(1,3)*l(1)^3 + Om1(1)*h(1,3)*h(1,4)*l(1)^3 + Om1(2)*h(2,1)*h(2,3)*l(1)^3 + Om1(2)*h(2,3)*h(2,4)*l(1)^3 + Om1(3)*h(3,1)*h(3,3)*l(1)^3 + Om1(3)*h(3,3)*h(3,4)*l(1)^3 + Om1(4)*h(4,1)*h(4,3)*l(1)^3 + Om1(4)*h(4,3)*h(4,4)*l(1)^3 + Om1(1)*h(1,1)*h(1,3)*l(1)*l(2)^2 - Om1(1)*h(1,3)*h(1,4)*l(1)*l(2)^2 + Om1(1)*h(1,1)*h(1,3)*l(1)*l(3)^2 + 2*Om1(1)*h(1,2)*h(1,3)*l(1)^2*l(3) - Om1(1)*h(1,3)*h(1,4)*l(1)*l(3)^2 + Om1(2)*h(2,1)*h(2,3)*l(1)*l(2)^2 - Om1(2)*h(2,3)*h(2,4)*l(1)*l(2)^2 + Om1(2)*h(2,1)*h(2,3)*l(1)*l(3)^2 + 2*Om1(2)*h(2,2)*h(2,3)*l(1)^2*l(3) - Om1(2)*h(2,3)*h(2,4)*l(1)*l(3)^2 + Om1(3)*h(3,1)*h(3,3)*l(1)*l(2)^2 - Om1(3)*h(3,3)*h(3,4)*l(1)*l(2)^2 + Om1(3)*h(3,1)*h(3,3)*l(1)*l(3)^2 + 2*Om1(3)*h(3,2)*h(3,3)*l(1)^2*l(3) - Om1(3)*h(3,3)*h(3,4)*l(1)*l(3)^2 + Om1(4)*h(4,1)*h(4,3)*l(1)*l(2)^2 - Om1(4)*h(4,3)*h(4,4)*l(1)*l(2)^2 + Om1(4)*h(4,1)*h(4,3)*l(1)*l(3)^2 + 2*Om1(4)*h(4,2)*h(4,3)*l(1)^2*l(3) - Om1(4)*h(4,3)*h(4,4)*l(1)*l(3)^2;
            theRoots = roots([4*coeff(4) 3*coeff(3) 2*coeff(2) coeff(1)]);
            theRoots = theRoots(imag(theRoots) == 0);
            oldL = l(4);
            l(4) = bestRoot(theRoots, coeff);
            [nextValue nextValueA] = evaluateObjective(g(:, n), h, l, V, m, covar, z(n, :), Om1, Om2, Om3, Om4);
%             if prevValue < nextValue
%                 badOpt = badOpt + nextValue - prevValue;
%                 l(4) = oldL; badOptCount = badOptCount + 1;
%             end
            prevValue = nextValue; prevValueA = nextValueA;
            allOptCount = allOptCount + 4;
            if (prevL - l)'*(prevL - l) < 1e-3
                break;
            end
        end
        X(:, n) = l;
        %% Upd 190509 New transform
        f(1, n) = (l(1)^2 + l(2)^2 + l(3)^2 + l(4)^2);
        f(2, n) = 2*l(1)*l(3);
        f(3, n) = 2*l(1)*l(4);
        f(4, n) = (l(1)^2 - l(2)^2 - l(3)^2 - l(4)^2);
        %%
    end
    %% V (noise covariance on observations g) %%
    if Hspflag == 0
        tempV = (g - H*f).*(g - H*f);
    else
        tempV = zeros(size(g));
        for n = 1:N
            tempV(:, n) = reshape(Hsp(:, n), [4 4]) * f(:, n);
            tempV = (g - tempV).*(g - tempV);
        end
    end
    V = (inv(N*Dinput) * sum(tempV(:)) + eps)*eye(Dinput);
    %% Compute restoration error %%
    tempE = (f - cheat).*(f - cheat);
    E = inv(N*D) * sum(tempE(:));
%     E = 0;
%     ISNR = 0;
    [junk ISNR] = computePolarimetricIndex(f(:), cheat(:), IN(:));
%    fprintf('#%d: V = %f, Error = %f, ISNR = %f, Failed opt ratio = %f, Av.fail.Value = %f, \n', iterations, V(1, 1), E, ISNR, badOptCount / allOptCount, badOpt/badOptCount);
    meanDOP = mean( sqrt(f(2, :).^2 + f(3, :).^2 + f(4, :).^2) ./ f(1, :) );
    fprintf('#%d: V = %f, Error = %f, ISNR = %f, mean DOP = %f\n', iterations, V(1, 1), E, ISNR, meanDOP);    
end
toc;
%% Upd190509: New transform
%% Permute lambda channels 2, 3, 4.
tt = X(2, :);
X(2, :) = X(3, :);
X(3, :) = X(4, :);
X(4, :) = tt; clear tt;
return;

function res = bestRoot(theRoots, coeff)
% This will return the root that gives the minimal value
% for coeff(4)*x^4 + coeff(3)*x^3 + coeff(2)*x^2 + coeff(1)*x^1
[junk I] = min(coeff(4) * theRoots.^4 + coeff(3) * theRoots.^3 + coeff(2) * theRoots.^2 + coeff(1) * theRoots);
res = theRoots(I);
return;

function [res res2] = evaluateObjective(g, H, l, V, m, covar, z, Om1, Om2, Om3, Om4)
% Cette fonction doit etre _minimise_.
% Cela est fait pendant le pas ou on resout les 4 polynomes
s = zeros(4, 1); K = size(m, 2);
%% Upd190509 New transform 190509
s(1) = (l(1)^2 + l(2)^2 + l(3)^2 + l(4)^2);
s(2) = 2*l(1)*l(3);
s(3) = 2*l(1)*l(4);
s(4) = (l(1)^2 - l(2)^2 - l(3)^2 - l(4)^2);
%% Upd 190509 New transform: h/2 au lieu de h
%% H allagh tou "H" einai plasmatiki: ginetai mono gia pragmatopoihthei o
%% neos metasxhmatismos programmatistika pio efkola.
H = H/2;
%%
res = (g - H*s)'*inv(V)*(g - H*s);
for j = 1:K
    res = res + (l - m(:, j))'*inv(covar(:,:,j))*(l - m(:, j)) * z(j);
end
e = H*s;
res2 = e'*diag(Om1)*e + l'*Om2*l + Om3*e + Om4*l;
return;

function res = checkStokeValidity(f)
if numel(f) <= 4
    if f(1) >= 0 && f(1)^2 >= f(2)^2 + f(3)^2 + f(4)^2
        res = 1;
        return;
    end
    res = 0;
    return;
else
    %Otherwise assume that f is 4xN
    res = (f(1, :) >= 0 & f(1, :).^2 >= ...
        f(2, :).^2 + f(3, :).^2 + f(4, :).^2);
end
return;

function [pos direction] = getNeighbourInfo(n)
switch n
    case {1}
        pos = [0 -1];
        direction = 1;
    case {2}
        pos = [-1 0];
        direction = 2;
    case {3}
        pos = [0 1];
        direction = 1;
    case {4}
        pos = [1 0];
        direction = 2;
    otherwise
        disp('Error: Unknown neighbour');
end
return;
function res = getDirectionInfo(d)
switch d
    case {1}
        res = [1 3];
    case {2}
        res = [2 4];
    otherwise
        disp('Error: Unknown direction');
end
return;

function res = getTotalNeighbours(offset, imageSize)
res = prod(imageSize - abs(offset));
return;

function res = vFunction(v, vConstant)
res = log(0.5*v) - psi(0.5*v) + vConstant;
return;

function res = solveQuad(a, b, c)
if a < 0
    a = -a;
    b = -b;
    c = -c;
end
determining = b.^2 - 4 * a .* c;
res = (-b + sqrt(determining)) ./ (2*a);
return;

function res = translation(rect, offset)
T = maketform('affine', [eye(2) [offset(1) offset(2)]'; 0 0 1]');
R = makeresampler('nearest', 'fill');
res = tformarray(rect, T, R, [1 2], [1 2], size(rect),[],[]);
return;