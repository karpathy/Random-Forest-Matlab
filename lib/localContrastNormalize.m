function y = localContrastNormalize(I, windowSize)
% Locally contrast normalizes an image. This is very often used as a
% pre-processing technique in feature learning. It is described in more
% detail in, for example,
%
% Jarett et al., "What is the Best Multi-Stage Architecture for Object
% Recognition?" (ICCV 2009)
%
%
% USAGE
%  y = localConstrastNormalize( I, [windowSize], [removeBorder] )
% 
% INPUTS
%  I             - Input image. Can be uint8 in [0 255] or double in [0 1], 
%                  greyscale or color.
%  windowSize    - [8] Size of the contrasting window 
%  
% OUTPUTS
%  y             - the normalized image
%
% EXAMPLE
%  G = rgb2gray(imread('data/lenna.jpg'));
%  y = localContrastNormalize(G);
%  figure(1); imshow(G,[]); figure(2); imshow(y,[]);
%

if(isa(I, 'uint8')), I = double(I) / 255.0; end
if nargin < 2 || isempty(windowSize), windowSize = 8; end

nd = ndims(I);

wSide = normpdf(linspace(-2, 2, windowSize), 0, 1);
w = wSide'*wSide;
w = w/sum(sum(w));

conv2opt= 'same'; 
if removeBorder, conv2opt= 'full'; end

if nd == 2
    
    % grayscale input
    v = I - conv2(I, w, conv2opt);
    s = sqrt(conv2(v.^2, w, conv2opt));
    c = mean(mean(s));
    y = v./max(c, s);
    
else
    
    % RGB input
    v1 = I(:,:,1) - conv2(I(:,:,1), w, conv2opt);
    v2 = I(:,:,2) - conv2(I(:,:,2), w, conv2opt);
    v3 = I(:,:,3) - conv2(I(:,:,3), w, conv2opt);
     
    s = sqrt(conv2(v1.^2, w, conv2opt) + conv2(v2.^2, w, conv2opt) + conv2(v3.^2, w, conv2opt));
    c = mean(mean(s));
    mm= max(c, s);
    y = cat(3, v1./mm, v2./mm, v3./mm);
    
end



end
