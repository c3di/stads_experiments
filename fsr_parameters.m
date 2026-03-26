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

function [fsr_parameters] = fsr_parameters(quality)

   
    fsr_parameters = struct();
    
    % default variables are set in this function.
    fsr_parameters.block_size = 16;
    fsr_parameters.conc_weighting = 0.5;
    fsr_parameters.debug = 0;
    fsr_parameters.rhos = [0.80, 0.70, 0.66, 0.64];
    fsr_parameters.threshold_stddev_Y = [0.014, 0.030, 0.090];
    fsr_parameters.threshold_stddev_Cx = [0.006, 0.010, 0.028];
    
    % quality profile dependent variables
    if (strcmp(quality, 'BEST'))
        fsr_parameters.block_size_min = 2;
        fsr_parameters.fft_size = 64;
        fsr_parameters.max_iter = 400;
        fsr_parameters.min_iter = 50;
        fsr_parameters.iter_const = 2000;
        fsr_parameters.orthogonality_correction = 0.2;
    elseif (strcmp(quality, 'COMPROMISE'))
        fsr_parameters.block_size_min = 2;
        fsr_parameters.fft_size = 32;
        fsr_parameters.max_iter = 100;
        fsr_parameters.min_iter = 20;
        fsr_parameters.iter_const = 2000;
        fsr_parameters.orthogonality_correction = 0.35;
    elseif (strcmp(quality, 'FAST'))
        fsr_parameters.block_size_min = 4;
        fsr_parameters.fft_size = 32;
        fsr_parameters.max_iter = 100;
        fsr_parameters.min_iter = 20;
        fsr_parameters.iter_const = 1000;
        fsr_parameters.orthogonality_correction = 0.5;
    else
        error('Unkown quality level set, supported: FAST, COMPROMISE, BEST');
    end
end
