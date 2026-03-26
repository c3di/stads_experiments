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

function [sigma_n] = fsr_standard_deviation(distorted_block_2d, error_mask_2d) 
    
    % compute variance
    undistorted_pixel = distorted_block_2d(error_mask_2d ~= 0);
    var_w = var(undistorted_pixel(:));

    % compute normalized standard deviation
    sigma_n = sqrt(var_w)/255;
    if (sigma_n < 0)
        sigma_n = 0;
    elseif (sigma_n > 1)
        sigma_n = 1;
    end
end
