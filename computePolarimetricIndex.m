function [index improvementIndex] = computePolarimetricIndex(estimate, groundTruth, degraded)
% res = computePolarimetricIndex(estimate, groundTruth, degraded)
%
% Computes an index of estimation success between
% an image estimate (restoration) and the real, ground-truth image.
% We used this index on the ICIP 2009 submission, in
% the context of polarimetric image restoration.
%
% If a third input is provided (degraded image)
% then alternatively the improvement on the image is computed.
% (Parallel to the ISNR value)
%
% All values are max 100% (best result)
%
% Examples:
%       [i1 i2] = computePolarimetricIndex(f7_1, convertJxN(S), convertJxN(G))
% The following
%       computePolarimetricIndex(S, S)
% Should return 1.
%
% G.Sfikas 14 Fev 2009

diff = (estimate - groundTruth);
denom = groundTruth + 1e-3;

diff = diff(:);
denom = denom(:);

normDiff = sqrt(sum(diff .* diff));
normDenom = sqrt(sum(denom .* denom));
index = 1 - normDiff ./ normDenom;

if exist('degraded', 'var') == 1
    degraded = degraded(:);
    diffDegraded = (degraded - denom);
    normDiffDegraded = sqrt(sum(diffDegraded .* diffDegraded));
%     improvementIndex = 1 - normDiff ./ normDiffDegraded;
    improvementIndex = 20 * log10(normDiffDegraded / normDiff);
end

return;