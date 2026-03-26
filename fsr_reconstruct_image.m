%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                  %
%   Copyright (c) 2018 by                                          %
%   Chair of Multimedia Communications and Signal Processing       %
%   Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)        %
%   - all rights reserved -                                        %
%                                                                  %
%   YOU ARE USING THIS PROGRAM AT YOUR OWN RISK! THE AUTHOR        %
%   IS NOT RESPONSIBLE FOR ANY DAMAGE OR DATA-LOSS CAUSED BY THE   %
%   USE OF THIS PROGRAM.                                           %
%                                                                  %
%                                                                  %
%   If you have any questions please contact:                      %
%                                                                  %
%   Nils Genser, M.Sc. or Dr.-Ing. Juergen Seiler                  %
%   Multimedia Communications and Signal Processing                %
%   University of Erlangen-Nuremberg                               %
%   Cauerstr. 7                                                    %
%   91058 Erlangen, Germany                                        %
%                                                                  %
%   email: { nils.genser, juergen.seiler } @ fau.de                %
%                                                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function reconstructed_img = fsr_reconstruct_image(sampled_img, sampling_mask, quality)
    
    type = 'CONNECTED_LOSSES';

    % checks
    if (length(size(sampled_img)) == 2) % grayscale image
        reconstructed_img = fsr_determine_processing_order(sampled_img, sampling_mask, quality, type, 'Y') / 255;
    elseif (length(size(sampled_img)) == 3 && size(sampled_img,3) == 3) % rgb image?
        image_YCbCr = rgb2ycbcr(sampled_img) .* sampling_mask;
        reconstructed_image_YCbCr(:,:,1) = fsr_determine_processing_order(image_YCbCr(:,:,1), sampling_mask, quality, type, 'Y');
        reconstructed_image_YCbCr(:,:,2) = fsr_determine_processing_order(image_YCbCr(:,:,2), sampling_mask, quality, type, 'Cx');
        reconstructed_image_YCbCr(:,:,3) = fsr_determine_processing_order(image_YCbCr(:,:,3), sampling_mask, quality, type, 'Cx');
        reconstructed_img = ycbcr2rgb(reconstructed_image_YCbCr/255);
    else
        error('Grayscale or RGB images supported, only!');
    end
end
