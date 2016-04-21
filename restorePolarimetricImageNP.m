function [f V z X] = restorePolarimetricImageNP(IN, H, cheat, K, cheatInit)
% f = restorePolarimetricImageNP(IN)
%
% Restore the distorted and noise-corrupted IN, to the original stokes 4-channel image.
% Input argument 1 (IN) and output (f) is a X x Y x 4 matrix.
% Second input is the distortion matrix (the H in Hf + n)
% "cheat" is the ground-truth correct image; this is used to calculate the estimated error.
% "cheatInit" is an optional user-provided initial estimate of the restored image.
% For the rest of the parameters, see the papers.
%
% Notes:
%      -NP on the title stands for "no parametrization". This refers to the gaussian mix
%       being imposed directly on the stokes vectors (f) as opposed to the
%       "restorePolarimetricImage" routine.
%      -The prior used is the Student-t continuous line process similar to
%       the one described in [1].
%
% @article{sfikas2011polar,
%  title={Recovery of polarimetric Stokes images by spatial mixture models},
%  author={Sfikas, Giorgos and Heinrich, Christian and Zallat, Jihad and Nikou, Christophoros and Galatsanos, Nikos},
%  journal={JOSA A},
%  volume={28},
%  number={3},
%  pages={465--474},
%  year={2011}
%}
%
% @inproceedings{sfikas2009polar,
% title={Joint recovery and segmentation of polarimetric images using a compound mrf and mixture modeling},
%  author={Sfikas, Giorgos and Heinrich, Christian and Zallat, Jihad and Nikou, Christophoros and Galatsanos, N},
%  booktitle={IEEE International Conference on Image Processing (ICIP)},
%  pages={3901--3904},
%  year={2009}
%}
%
% G.Sfikas 21 Apr 2016
flagGridScan = 0;
%%Compute quelques quantites necessaires
if(exist('K', 'var') == 0)
    K = 7;
end
D = 4;
NhoodSize = 4;
NhoodDirections = 2; %WARNING:Unreliable constant (do not change this)
% Initialization
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
tempE = (f - cheat).*(f - cheat);
E = inv(N*D) * sum(tempE(:));
fprintf('-- Initial error (using H^-1 * g) -- : %f\n', E);
[junk ISNR] = computePolarimetricIndex(f(:), cheat(:), IN(:));
fprintf('-- Initial improvement over SNR (using H^-1 * g) -- : %f\n', ISNR);
fprintf('-- Admissible vectors: %d out of %d.\n', sum(checkStokeValidity(f)), N);
%%Initialization
X = f;
allDataCov = cov(X');
covar = zeros(D, D, K);
% Deterministic initialization -- K-means
% disp('Warning: Deterministic k-means is disabled!');
[m w] = deterministicKmeans(X(:,1:N), K);
w = w'*ones(1, N);
% w = w + 0.5*rand(K, N); % Add some noise..
w = (w ./ (ones(K, 1) * (sum(w, 1)+eps)))';
for i = 1:K
    % Make each std.deviation equal to 1/K of total std.deviation.
    covar(:,:,i) = allDataCov / K^2 + eps*eye(D);
end
%%% FIN - Deterministic initialization -- K-means %%%
beta = double(0.8 * ones(NhoodDirections, K));
u = double(ones(NhoodSize, K, N));
logU = zeros(NhoodSize, K, N);
v = double(ones(NhoodDirections, K));
%%% Hyperparameter initialization
z = inv(N) * ones(N, K);
wDiff2 = zeros(NhoodSize, K, N);
wDiff2Beta = zeros(NhoodSize, K, N);
V = 1e-6*eye(Dinput);
%
% This is used for the MC amelioration step.
randomData = randn(4, 1000);
%
% Main EM loop
%
tic
disp('Warning: Disabled MC Monte Carlo amelioration for speed.');
fprintf('Restoration using Student-t continuous LP, gmm assumed on Stokes vectors\n');
fprintf('------------------------------------------------------------------------');
for iterations = 1:7
    likelihood = 0;
    % E-step
    for i = 1:K
        %
        % UPDATE 11 SEP 2010.
        % Have to add "inverse(Xi)" on updates (see paper), since we're
        % working with _Truncated_ Gaussians.
        % Part of the code is copied from the "mu"&"Sigma" update section
        % below.
        currentRandomData = chol(covar(:,:,i))' * randomData + m(:, i) * ones(1, size(randomData, 2));
        constr1 = currentRandomData(1, :);
        constr2 = currentRandomData(1, :).^2 - currentRandomData(2, :).^2 - currentRandomData(3, :).^2 - currentRandomData(4, :).^2;
        Xi = sum(constr1 >= 0 & constr2 >= 0) / size(randomData, 2);
 %       fprintf('%f ', Xi);
        z(:, i) = inv(Xi) * w(:, i) .* gaussianValue(X, m(:, i), covar(:,:,i)) + eps;
        likelihood = likelihood + z(:, i);        
    end
 %   fprintf('\n');
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
        newCovar = newCovar + (1e-5)*eye(D);
        if rcond(newCovar) > 1e3 * eps && det(newCovar) > 1e-20
            covar(:, :, i) = newCovar;
        end
        clear newCovar;
    end
    % Amelioration of means & covariances
    % ===================================
    % On the previous step we had hypothesized that the gaussian
    % assumed on the stokes vectors has a normalization constant like
    % sqrt(2*pi)*abs(st.dev); but this is mistaken since the stokes
    % obey a certain constraint on their values. In other words the true
    % stokes distribution is a truncated Gaussian.
    for j = 1:K
%%         currentRandomData = chol(covar(:,:,j))' * randomData + m(:, j) * ones(1, size(randomData, 2));
%%         constr1 = currentRandomData(1, :);
%%         constr2 = currentRandomData(1, :).^2 - currentRandomData(2, :).^2 - currentRandomData(3, :).^2 - currentRandomData(4, :).^2;
%%         fprintf('%f ', sum(constr1 >= 0 & constr2 >= 0) / size(randomData, 2));
%%         realZ = (sum(constr1 >= 0 & constr2 >= 0) / size(randomData, 2)) * (2*pi*det(covar(:,:,j)))^(-0.5);
         %
         L = chol(covar(:,:,j))';
         x = fminsearch(@(x) objectiveAmel(x, X, z(:, j), randomData), [m(:, j)' L(1:4, 1)' L(2:4, 2)' L(3:4, 3)' L(4:4, 4)'], ...
             optimset('Display', 'off'));
         newL = zeros(4, 4);
         newM = x(1:4);
         newL(1:4, 1) = x((1:4) + 4);
         newL(2:4, 2) = x((5:7) + 4);
         newL(3:4, 3) = x((8:9) + 4);
         newL(4:4, 4) = x((10) + 4);
         if sum(diag(newL) <= 0) > 0
             fprintf('restorePolarimetricImageNP: Hooke-Jeeves/Monte Carlo step: covariance computed as non-positive definite (reject)\n');
             continue;
         end        
         newCovar = newL * newL';
         if rcond(newCovar) > 1e3 * eps && det(newCovar) > 1e-20
             tt = abs(m(:, j)' - newM);
             fprintf('restorePolarimetricImageNP: HJ/MC step: new mean offset for MEAN: %f\n', sum(tt(:)) / numel(newM));
             tt = abs(covar(:, :, j) - newCovar);
             fprintf('restorePolarimetricImageNP: HJ/MC step: new mean offset for COV: %f\n', sum(tt(:)) / numel(newCovar));            
             m(:, j) = newM;            
             covar(:, :, j) = newCovar;
         end        
    end
    % Compute q*(U) %%
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
    % M-step: f
    % Compute _valid_ (affirming to constraints) stoke vector estimates.
    %
    for n = 1:N
        if Hspflag == 1
            H = reshape(Hsp(:, n), [4 4]);
        end
        coeffA = H'*inv(V)*H;
        coeffB = g(:, n)' * inv(V) * H;
        for j = 1:K
            coeffA = coeffA + z(n, j) * inv(covar(:, :, j) + eps);
            coeffB = coeffB + z(n, j) * m(:, j)' * inv(covar(:, :, j) + eps);
        end
        f(:, n) = computeStokesEstimate(amelioreStoke(pinv(H)*g(:, n)), coeffA, coeffB);
    end
    X = f;
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
    fprintf('#%d: V = %f, Error = %f, ISNR = %f\n', iterations, V(1, 1), E, ISNR);
end
toc
return;

function res = objectiveAmel(x, X, z, randomData)
L = zeros(4, 4);
m = x(1:4); m = m';
L(1:4, 1) = x((1:4) + 4);
L(2:4, 2) = x((5:7) + 4);
L(3:4, 3) = x((8:9) + 4);
L(4:4, 4) = x((10) + 4);
covar = L * L';
%
currentRandomData = L * randomData + m * ones(1, size(randomData, 2));
constr1 = currentRandomData(1, :);
constr2 = currentRandomData(1, :).^2 - currentRandomData(2, :).^2 - currentRandomData(3, :).^2 - currentRandomData(4, :).^2;
% It is realZ = a * (2pi*det(covar))^(-0.5)
% so minlogRealZ defined as -2log(realZ) is
% -2log(a) + logdet(covar) + const.
minlogRealZ = -2 * log(sum(constr1 >= 0 & constr2 >= 0) / size(randomData, 2)) + logdet(covar);
% and remains 
res = (minlogRealZ + mahalanobis(X, m, covar))' * z;
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

function res = amelioreStoke(f)
res = f;
if checkStokeValidity(f) == 1
    return;
end
res(1) = sqrt(res(2)^2 + res(3)^2 + res(4)^2);
return;

function res = checkStokeValidity(f)
if f(1) >= 0 && f(1)^2 >= f(2)^2 + f(3)^2 + f(4)^2
    res = 1;
    return;
end
res = 0;
return;

function res = computeStokesEstimate(init, A, b)
f = init(:); b = b(:);
for iter = 1:10
    h = 1.0;
    % At least one valide Stoke v. must be found, for h = 0.
    while 1 
        if h < 0
            break;
        end
        x = (1 - h)*f + h*(inv(A)*b);
        if checkStokeValidity(x) == 1
            f = x;
            if h == 1
                % This is as best we can do, end
                res = x;
                return;
            end
            break;
        end
        h = h - 0.1;
    end
    % Now make a random step.. in a scale of 0.1
    prevValue = objectiveFunction(f, A, b);
    for i = 1:5
        newf = amelioreStoke(f + 0.1*randn(4, 1));
        newValue = objectiveFunction(newf, A, b);
        if newValue < prevValue
            f = newf;
            break;
        end
    end
end
res = f;
return;

function res = objectiveFunction(f, A, b)
res = 0.5*f'*A*f - b'*f;
return;
