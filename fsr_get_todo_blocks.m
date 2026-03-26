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

function [set_later, set_now, set_process_this_block_size, sigma_n_array] = fsr_get_todo_blocks(sampled_img, sampling_mask, set_todo, block_size, block_size_min, border_width, fft_size, homo_threshold, rho_homo, blocks_per_column, blocks_per_line, debug)
    
    % declare variables and constants
    set_now = [];
    set_later = [];
    set_process_this_block_size = zeros(blocks_per_column, blocks_per_line);
    sigma_n_array = zeros(blocks_per_column, blocks_per_line);
    list_length = size(set_todo, 1);
    img_height = size(sampled_img, 1);
    img_width = size(sampled_img, 2);
    reconstructed_img = sampled_img;
    blocks_quadernary = 0:3;
    
    % DEBUG BEGIN
    % calculate sigma_n_min for other block sizes, as well
    if (debug == 1) 
        blocks_per_column = ceil(size(sampled_img,1)/block_size);
        blocks_per_line = ceil(size(sampled_img,2)/block_size);
        sigma_n_max = 0;
        sigma_n_min = 1;
        
         for yblock_counter = 0:blocks_per_column-1
             for xblock_counter = 0:blocks_per_line-1
                if prod(prod(sampling_mask(min(img_height,yblock_counter*block_size+(1:block_size)), min(img_width, xblock_counter*block_size+(1:block_size))))) == 0
                    % calculation of the extrapolation area's borders
                    left_border = min(xblock_counter*block_size, border_width);
                    top_border = min(yblock_counter*block_size, border_width);
                    right_border = max(0, min(img_width-(xblock_counter+1)*block_size, border_width));
                    bottom_border = max(0, min(img_height-(yblock_counter+1)*block_size, border_width));
                   
                    % extract blocks from images
                    distorted_block_2d = reconstructed_img((yblock_counter*block_size-top_border+1):min(img_height,(yblock_counter*block_size+block_size+bottom_border)), (xblock_counter*block_size-left_border+1):min(img_width,(xblock_counter*block_size+block_size+right_border)));
                    error_mask_2d = sampling_mask((yblock_counter*block_size-top_border+1):min(img_height,(yblock_counter*block_size+block_size+bottom_border)), (xblock_counter*block_size-left_border+1):min(img_width,(xblock_counter*block_size+block_size+right_border)));
                   
                    % determine normalized and weighted standard deviation
                    if(block_size > block_size_min)
                        sigma_n  = standard_deviation_weighted_nils(distorted_block_2d, error_mask_2d, fft_size, rho_homo);
                        if (sigma_n > sigma_n_max) 
                            sigma_n_max = sigma_n;
                        end
                        if (sigma_n < sigma_n_min)
                            sigma_n_min = sigma_n; 
                        end
                    end
                end 
             end
         end
    end
    % DEBUG END
    
    % calculate block lists
    for entry = 0:list_length-1
        xblock_counter = set_todo(entry+1, 1);
        yblock_counter = set_todo(entry+1, 2);
        
        left_border = min(xblock_counter*block_size, border_width);
        top_border = min(yblock_counter*block_size, border_width);
        right_border = max(0, min(img_width-(xblock_counter+1)*block_size, border_width));
        bottom_border = max(0, min(img_height-(yblock_counter+1)*block_size, border_width));
       
        % extract blocks from images
        distorted_block_2d = reconstructed_img((yblock_counter*block_size-top_border+1):min(img_height,(yblock_counter*block_size+block_size+bottom_border)), (xblock_counter*block_size-left_border+1):min(img_width,(xblock_counter*block_size+block_size+right_border)));
        error_mask_2d = sampling_mask((yblock_counter*block_size-top_border+1):min(img_height,(yblock_counter*block_size+block_size+bottom_border)), (xblock_counter*block_size-left_border+1):min(img_width,(xblock_counter*block_size+block_size+right_border)));
       
        % determine normalized and weighted standard deviation
        if (block_size > block_size_min)
            sigma_n = fsr_standard_deviation(distorted_block_2d, error_mask_2d);
            sigma_n_array(yblock_counter+1, xblock_counter+1) = sigma_n;
          
            % homogeneous case
            if (sigma_n < homo_threshold) 
                set_now(end+1, 1) = xblock_counter;
                set_now(end, 2) = yblock_counter;
                set_process_this_block_size(yblock_counter+1, xblock_counter+1) = 255;
                
                % DEBUG BEGIN
                if (debug == 1)
                    hold on;
                    edge_color = ((sigma_n-sigma_n_min)/(homo_threshold-sigma_n_min));
                    if (edge_color < 0)
                        edge_color = 0;
                    elseif (edge_color > 1)
                        edge_color = 1;
                    end
                    rectangle('position',[xblock_counter*block_size+0.5 yblock_counter*block_size+0.5 block_size block_size], ...
                    'EdgeColor',[1 edge_color 1], 'LineStyle','-')
                end 
                % DEBUG END
            else % heterogeneous case
                % DEBUG BEGIN
                if (debug == 1)
                    hold on;
                    rectangle('position',[xblock_counter*block_size+0.5 yblock_counter*block_size+0.5 block_size block_size], ...
                    'EdgeColor',[0 ((sigma_n_max-sigma_n)/(sigma_n_max-homo_threshold)) 1], 'LineStyle','-') 
                end
                % DEBUG END
                
                yblock_counter_quadernary = yblock_counter*2;
                xblock_counter_quadernary = xblock_counter*2;
                for quader_counter = blocks_quadernary
                    if quader_counter == 0
                        yblock_offset = 0;
                        xblock_offset = 0;
                    elseif quader_counter == 1
                        yblock_offset = 0;
                        xblock_offset = 1;
                    elseif quader_counter == 2
                        yblock_offset = 1;
                        xblock_offset = 0;
                    elseif quader_counter == 3
                        yblock_offset = 1;
                        xblock_offset = 1;
                    end
                    set_later(end+1, 1) = xblock_counter_quadernary+xblock_offset;
                    set_later(end, 2) = yblock_counter_quadernary+yblock_offset;
                end 
            end 
        end
    end
end
