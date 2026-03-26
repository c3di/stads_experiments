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

function extrapolated_block = fsr_extrapolate_block(distorted_block, error_mask, fsr_parameters, rho, normedStdDev)
    
    % parameters
    fft_size = fsr_parameters.fft_size;
    orthogonality_correction = fsr_parameters.orthogonality_correction;
    [M, N] = size(distorted_block);
    fft_x_offset = floor((fft_size-N)/2);
    fft_y_offset = floor((fft_size-M)/2);

    % weighting function
    w = zeros(fft_size);
    w(fft_y_offset+(1:M), fft_x_offset+(1:N)) = error_mask;
    for u = 0:fft_size-1
        for v = 0:fft_size-1
            w(u+1,v+1) = w(u+1,v+1) * rho^(sqrt((u+0.5-(fft_y_offset+M/2))^2 + (v+0.5-(fft_x_offset+N/2))^2));
        end
    end
    W = fft2(w);
    W_padded = [W, W; W, W];
    
    % frequency weighting
    frequency_weighting = ones(fft_size, fft_size/2+1);
    for y=0:fft_size-1
        for x=0:fft_size/2
            y2 = fft_size/2 - abs(y - fft_size/2);
            x2 = fft_size/2 - abs(x - fft_size/2);
            frequency_weighting(y+1, x+1) = 1 - sqrt(x2*x2 + y2*y2)*sqrt(2)/fft_size;
        end
    end

    % pad image to fft window size
    f = zeros(fft_size);
    f(fft_y_offset+(1:M), fft_x_offset+(1:N)) = distorted_block;

    % create initial model
    G = zeros(fft_size);

    % calculate initial residual
    Rw = fft2(f.*w);
    Rw = Rw(1:fft_size, 1:fft_size/2+1);
    
    % estimate ideal number of iterations (GenserIWSSIP2017)
    % calculate stddev if not available (e.g., for smallest block size)
    if (normedStdDev == 0)
        normedStdDev = fsr_standard_deviation(distorted_block, error_mask) ;
    end
    num_iters = round(fsr_parameters.iter_const * normedStdDev);
    if (num_iters < fsr_parameters.min_iter)
        num_iters = fsr_parameters.min_iter;
    elseif (num_iters > fsr_parameters.max_iter)
		num_iters = fsr_parameters.max_iter;
    end
    
    iter_counter = 0;
    while (iter_counter < num_iters) % Spectral Constrained FSE (GenserIWSSIP2018)
        projection_distances = abs(Rw(:)) .* frequency_weighting(:);
        [~, bf2select] = max(projection_distances);
        bf2select = bf2select(1)-1;
        v = floor(bf2select/fft_size);
        u = mod(bf2select,fft_size);
        
        % exclude second half of first and middle col
        if (v == 0 && u > fft_size/2 || v == fft_size/2 && u > fft_size/2)
            u_prev = u;
            u = fft_size-u;
            Rw(u+1,v+1) = conj(Rw(u_prev+1,v+1));
        end

        % calculate complex conjugate solution
        u_cj = -1; v_cj = -1;
        % fill first lower col (copy from first upper col)
        if (u >= 1 && u < fft_size/2 && v == 0)
            u_cj = fft_size-u;
            v_cj = v;
        end
        % fill middle lower col (copy from first middle col)
        if (u >= 1 && u < fft_size/2 && v == fft_size/2)
            u_cj = fft_size-u;
            v_cj = v;
        end
        % fill first row right (copy from first row left)
        if (u == 0 && v >= 1 && v < fft_size/2)
            u_cj = u;
            v_cj = fft_size-v;
        end
        % fill middle row right (copy from middle row left)
        if (u == fft_size/2 && v >= 1 && v < fft_size/2)
            u_cj = u;
            v_cj = fft_size-v;
        end
        % fill cell upper right (copy from lower cell left)
        if (u >= fft_size/2+1 && v >= 1 && v < fft_size/2)
            u_cj = fft_size-u;
            v_cj = fft_size-v;
        end
        % fill cell lower right (copy from upper cell left)
        if (u >= 1 && u < fft_size/2 && v >= 1 && v < fft_size/2)
            u_cj = fft_size-u;
            v_cj = fft_size-v;
        end
        
        % add coef to model and update residual
        if (u_cj ~= -1 && v_cj ~= -1)
            expansion_coefficient = orthogonality_correction * Rw(u+1, v+1) / W(1);
            G(u+1, v+1) = G(u+1, v+1) + fft_size^2 * expansion_coefficient;
            G(u_cj+1, v_cj+1) = conj(G(u+1, v+1));
            Rw = Rw -  expansion_coefficient * W_padded(fft_size-u+1:2*fft_size-u, fft_size-v+1:fft_size-v+1+fft_size/2) ...
                    -  conj(expansion_coefficient) * W_padded(fft_size-u_cj+1:2*fft_size-u_cj, fft_size-v_cj+1:fft_size-v_cj+1+fft_size/2);
            iter_counter = iter_counter + 1; % ... as two basis functions were added
        else
            expansion_coefficient = orthogonality_correction * Rw(u+1, v+1) / W(1);
            G(u+1, v+1) = G(u+1, v+1) + fft_size^2 * expansion_coefficient;
            Rw = Rw -  expansion_coefficient * W_padded(fft_size-u+1:2*fft_size-u, fft_size-v+1:fft_size-v+1+fft_size/2);
        end
        
        iter_counter = iter_counter + 1;
    end

    % get pixels from model
    g = ifft2(G);

    % extract reconstructed pixels
    extrapolated_block = real(g(fft_y_offset+(1:M), fft_x_offset+(1:N)));
    orig_samples = find(error_mask~=0);
    extrapolated_block(orig_samples) = distorted_block(orig_samples);
end
