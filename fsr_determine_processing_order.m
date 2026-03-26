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

function [reconstructed_img] = fsr_determine_processing_order(sampled_img, sampling_mask, quality, loss_type, channel)
    
    % get required FSR default parameters
    fsr_parameters = fsr_parameters(quality);
    block_size = fsr_parameters.block_size;
    block_size_max = fsr_parameters.block_size;
    block_size_min = fsr_parameters.block_size_min;
    conc_weighting = fsr_parameters.conc_weighting;
    debug = fsr_parameters.debug;
    fft_size = fsr_parameters.fft_size;
    rho = fsr_parameters.rhos(1);
    sampled_img = double(sampled_img);
    reconstructed_img = double(sampled_img);
    sampling_mask =  double(max(0,sign(sampling_mask)));

    % read stddev LUT (GenserPCS2018)
    if strcmp(channel, 'Y')
        threshold_stddev_LUT = fsr_parameters.threshold_stddev_Y;
    elseif strcmp(channel, 'Cx')
        threshold_stddev_LUT = fsr_parameters.threshold_stddev_Cx;
    else
        error('Channel type unsupported!');
    end

    % optimized signal and loss geometry aware processing order (GenserPCS2018)
    if strcmp(loss_type, 'CONNECTED_LOSSES')
        threshold_stddev = threshold_stddev_LUT(1);

        set_later = [];
        img_height = size(sampled_img,1);
        img_width = size(sampled_img,2);

        % debugging ----------------
        if (debug == 1)
            figure('Name', 'original image');
            imshow(uint8(sampled_img));
            drawnow;
            refresh;
        end
        % --------------------------

        % initial scan of distorted blocks
        set_todo = [];
        for y = 0:ceil(size(sampled_img,1)/block_size)-1
            for x = 0:ceil(size(sampled_img,2)/block_size)-1
                if prod(prod(sampling_mask(min(img_height,y*block_size+(1:block_size)), min(img_width, x*block_size+(1:block_size))))) == 0
                    set_todo(end+1, 1) = x;
                    set_todo(end, 2) = y;
                end
            end
        end

        % loop over all distorted blocks and extrapolate them depending on
        % their block size
        while (block_size >= block_size_min)
            % debugging -----------
            if (debug == 1)
                figure('Name', 'image with errors and homogeneity - current iteration');
                cur_rec = reconstructed_img;
                imshow(double(cur_rec/255));
                block_size
                drawnow;
                refresh;
            end
            %----------------------

            blocks_per_column = ceil(size(sampled_img,1)/block_size);
            blocks_per_line = ceil(size(sampled_img,2)/block_size);
            nen_array = zeros(blocks_per_column, blocks_per_line);
            proc_array = zeros(blocks_per_column, blocks_per_line);
            block_list = zeros(blocks_per_column*blocks_per_line,2);
            sigma_n_array = zeros(blocks_per_column, blocks_per_line);

            if block_size > block_size_min
                if block_size < block_size_max
                    set_todo = set_later;
                end
                border_width = floor(fft_size-block_size)/2;
                [set_later, ~, set_process_this_block_size, sigma_n_array] = fsr_get_todo_blocks(sampled_img, sampling_mask, set_todo, block_size, block_size_min, border_width, fft_size, threshold_stddev, 1, blocks_per_column, blocks_per_line, debug);
            else
                set_process_this_block_size = ones(blocks_per_column, blocks_per_line);
                set_process_this_block_size = set_process_this_block_size .* 255;
            end

            %  if block to be extrapolated, increase nen of neighboring blocks
            for yblock_counter = 0:blocks_per_column-1
                for xblock_counter = 0:blocks_per_line-1
                    if prod(prod(sampling_mask(min(img_height,yblock_counter*block_size+(1:block_size)), min(img_width, xblock_counter*block_size+(1:block_size))))) == 0
                        if(yblock_counter>0 && xblock_counter>0)
                            nen_array(yblock_counter-1 +1, xblock_counter-1 +1) = nen_array(yblock_counter-1 +1, xblock_counter-1 +1) + 1;
                        end
                        if(yblock_counter>0)
                            nen_array(yblock_counter-1 +1, xblock_counter +1) = nen_array(yblock_counter-1 +1, xblock_counter +1) + 1;
                        end
                        if(yblock_counter>0 && xblock_counter<blocks_per_line-1)
                            nen_array(yblock_counter-1 +1, xblock_counter+1 +1) = nen_array(yblock_counter-1 +1, xblock_counter+1 +1) + 1;
                        end
                        if(xblock_counter>0)
                            nen_array(yblock_counter +1, xblock_counter-1 +1) = nen_array(yblock_counter +1, xblock_counter-1 +1) + 1;
                        end
                        if(xblock_counter<blocks_per_line-1)
                            nen_array(yblock_counter +1, xblock_counter+1 +1) = nen_array(yblock_counter +1, xblock_counter+1 +1) + 1;
                        end
                        if(yblock_counter<blocks_per_column-1 && xblock_counter>0)
                            nen_array(yblock_counter+1 +1,xblock_counter-1 +1) = nen_array(yblock_counter+1 +1,xblock_counter-1 +1) + 1;
                        end
                        if(yblock_counter<blocks_per_column-1)
                            nen_array(yblock_counter+1 +1, xblock_counter +1) = nen_array(yblock_counter+1 +1, xblock_counter +1) + 1;
                        end
                        if(yblock_counter<blocks_per_column-1 && xblock_counter<blocks_per_line-1)
                            nen_array(yblock_counter+1 +1, xblock_counter+1 +1) = nen_array(yblock_counter+1 +1, xblock_counter+1 +1) + 1;
                        end
                    end
                end
            end

            % determine if block itself has to be extrapolated
            for yblock_counter = 0:blocks_per_column-1
                for xblock_counter = 0:blocks_per_line-1
                    if prod(prod(sampling_mask(min(img_height, yblock_counter*block_size+(1:block_size)), min(img_width, xblock_counter*block_size+(1:block_size))))) ~= 0
                        nen_array(yblock_counter+1, xblock_counter+1) = -1;
                    else
                        if(yblock_counter==0 && xblock_counter==0)
                            nen_array(yblock_counter+1, xblock_counter+1) = nen_array(yblock_counter+1, xblock_counter+1) + 5;
                        end
                        if(yblock_counter==0 && xblock_counter==blocks_per_line-1)
                            nen_array(yblock_counter+1, xblock_counter+1) = nen_array(yblock_counter+1, xblock_counter+1) + 5;
                        end
                        if(yblock_counter==blocks_per_column-1 && xblock_counter==0)
                            nen_array(yblock_counter+1, xblock_counter+1) = nen_array(yblock_counter+1, xblock_counter+1) + 5;
                        end
                        if(yblock_counter==blocks_per_column-1 && xblock_counter==blocks_per_line-1)
                            nen_array(yblock_counter+1, xblock_counter+1) = nen_array(yblock_counter+1, xblock_counter+1) + 5;
                        end
                        if(yblock_counter==0 && xblock_counter~=0 && xblock_counter~=blocks_per_line-1)
                            nen_array(yblock_counter+1, xblock_counter+1) = nen_array(yblock_counter+1, xblock_counter+1) + 3;
                        end
                        if(yblock_counter==blocks_per_column-1 && xblock_counter~=0 && xblock_counter~=blocks_per_line-1)
                            nen_array(yblock_counter+1, xblock_counter+1) = nen_array(yblock_counter+1, xblock_counter+1) + 3;
                        end
                        if(yblock_counter~=0 && yblock_counter~=blocks_per_column-1 && xblock_counter==0)
                            nen_array(yblock_counter+1, xblock_counter+1) = nen_array(yblock_counter+1, xblock_counter+1) + 3;
                        end
                        if(yblock_counter~=0 && yblock_counter~=blocks_per_column-1 && xblock_counter==blocks_per_line-1)
                            nen_array(yblock_counter+1, xblock_counter+1) = nen_array(yblock_counter+1, xblock_counter+1) + 3;
                        end
                    end
                end
            end

            % if all blocks have 8 not extrapolated neighbors, penalize nen of blocks without any known samples by one
            if min(nen_array(:)) == 8
                for yblock_counter = 0:blocks_per_column-1
                    for xblock_counter = 0:blocks_per_line-1
                        if sum(sum(sampling_mask(min(img_height, yblock_counter*block_size+(1:block_size)), min(img_width, xblock_counter*block_size+(1:block_size))))) == 0
                            nen_array(yblock_counter+1, xblock_counter+1) = nen_array(yblock_counter+1, xblock_counter+1) + 1;
                        end
                    end
                end
            end

            % do actual processing per block
            all_blocks_finished = 0;
            while all_blocks_finished == 0
                % clear proc_array
                proc_array(:) = 1;

                % determin blocks to extrapolate
                min_nen = 99;
                bl_counter = 0;
                % add all homogeneous blocks that shall be processed to list
                % using same priority
                % begins with highest priority or lowest nen_array value
                for yblock_counter = 0:blocks_per_column-1
                    for xblock_counter = 0:blocks_per_line-1
                        % decisision whether block contains errors
                        if (nen_array(yblock_counter+1, xblock_counter+1) >= 0 && nen_array(yblock_counter+1, xblock_counter+1) < min_nen && set_process_this_block_size(yblock_counter+1, xblock_counter+1) == 255)
                            bl_counter = 0;
                            min_nen = nen_array(yblock_counter+1, xblock_counter+1);
                            proc_array(:) = 1;
                        end

                        if(nen_array(yblock_counter+1, xblock_counter+1) == min_nen && proc_array(yblock_counter+1, xblock_counter+1)~=0 && set_process_this_block_size(yblock_counter+1, xblock_counter+1) == 0)
                            nen_array(yblock_counter+1, xblock_counter+1) = -1;
                        end

                        if(nen_array(yblock_counter+1, xblock_counter+1) == min_nen && proc_array(yblock_counter+1, xblock_counter+1)~=0 && set_process_this_block_size(yblock_counter+1, xblock_counter+1) ~= 0)
                            block_list(bl_counter+1,1) = yblock_counter;
                            block_list(bl_counter+1,2) = xblock_counter;
                            bl_counter = bl_counter+1;

                            % block neighboring blocks from processing
                            if(yblock_counter>0 && xblock_counter>0)
                                proc_array(yblock_counter-1 +1, xblock_counter-1 +1) = 0;
                            end
                            if(yblock_counter>0)
                                proc_array(yblock_counter-1 +1,xblock_counter +1) = 0;
                            end
                            if(yblock_counter>0 && xblock_counter<blocks_per_line-1)
                                proc_array(yblock_counter-1 +1, xblock_counter+1 +1) = 0;
                            end
                            if(xblock_counter>0)
                                proc_array(yblock_counter +1, xblock_counter-1 +1) = 0;
                            end
                            if(xblock_counter<blocks_per_line-1)
                                proc_array(yblock_counter +1, xblock_counter+1 +1) = 0;
                            end
                            if(yblock_counter<blocks_per_column-1 && xblock_counter>0)
                                proc_array(yblock_counter+1 +1, xblock_counter-1 +1) = 0;
                            end
                            if(yblock_counter<blocks_per_column-1)
                                proc_array(yblock_counter+1 +1, xblock_counter +1) = 0;
                            end
                            if(yblock_counter<blocks_per_column-1 && xblock_counter<blocks_per_line-1)
                                proc_array(yblock_counter+1 +1, xblock_counter+1 +1) = 0;
                            end
                        end
                    end
                end

                max_bl_counter = bl_counter;
                block_list(bl_counter+1,1) = -1;
                block_list(bl_counter+1,2) = -1;
                if(bl_counter == 0)
                    all_blocks_finished = 1;
                end

                % blockwise extrapolation of all blocks that can be processed in parallel
                for bl_counter = 0:max_bl_counter-1
                    yblock_counter = block_list(bl_counter+1,1);
                    xblock_counter = block_list(bl_counter+1,2);

                    % calculation of the extrapolation area's borders
                    left_border = min(xblock_counter*block_size, border_width);
                    top_border = min(yblock_counter*block_size, border_width);
                    right_border = max(0, min(img_width-(xblock_counter+1)*block_size, border_width));
                    bottom_border = max(0, min(img_height-(yblock_counter+1)*block_size, border_width));

                    % extract blocks from images
                    distorted_block_2d = reconstructed_img((yblock_counter*block_size-top_border+1):min(img_height,(yblock_counter*block_size+block_size+bottom_border)), (xblock_counter*block_size-left_border+1):min(img_width,(xblock_counter*block_size+block_size+right_border)));
                    error_mask_2d = sampling_mask((yblock_counter*block_size-top_border+1):min(img_height,(yblock_counter*block_size+block_size+bottom_border)), (xblock_counter*block_size-left_border+1):min(img_width,(xblock_counter*block_size+block_size+right_border)));

                    % get actual stddev value as it is needed to estimate the
                    % best number of iterations
                    sigma_n_a = sigma_n_array(yblock_counter+1, xblock_counter+1);

                    % actual extrapolation
                    extrapolated_block_2d = fsr_extrapolate_block(distorted_block_2d, error_mask_2d, fsr_parameters, rho, sigma_n_a);

                    % update image and mask
                    reconstructed_img((yblock_counter*block_size+1):min(img_height,(yblock_counter+1)*block_size), (xblock_counter*block_size+1):min(img_width,(xblock_counter+1)*block_size)) = extrapolated_block_2d(top_border+1:end-bottom_border, left_border+1:end-right_border);
                    sampling_mask((yblock_counter*block_size+1):min(img_height,(yblock_counter+1)*block_size), (xblock_counter*block_size+1):min(img_width,(xblock_counter+1)*block_size)) = error_mask_2d(top_border+1:end-bottom_border, left_border+1:end-right_border) + (1-sign(error_mask_2d(top_border+1:end-bottom_border, left_border+1:end-right_border)))*conc_weighting;

                    % update nen-array
                    nen_array(yblock_counter +1, xblock_counter +1) = -1;
                    if(yblock_counter>0 && xblock_counter>0)
                        nen_array(yblock_counter-1 +1, xblock_counter-1 +1) = nen_array(yblock_counter-1 +1, xblock_counter-1 +1) - 1;
                    end
                    if(yblock_counter>0)
                        nen_array(yblock_counter-1 +1, xblock_counter +1) = nen_array(yblock_counter-1 +1, xblock_counter +1) - 1;
                    end
                    if(yblock_counter>0 && xblock_counter<blocks_per_line-1)
                        nen_array(yblock_counter-1 +1, xblock_counter+1 +1) = nen_array(yblock_counter-1 +1, xblock_counter+1 +1) - 1;
                    end
                    if(xblock_counter>0)
                        nen_array(yblock_counter +1, xblock_counter-1 +1) = nen_array(yblock_counter +1, xblock_counter-1 +1) - 1;
                    end
                    if(xblock_counter<blocks_per_line-1)
                        nen_array(yblock_counter +1, xblock_counter+1 +1) = nen_array(yblock_counter +1, xblock_counter+1 +1) - 1;
                    end
                    if(yblock_counter<blocks_per_column-1 && xblock_counter>0)
                        nen_array(yblock_counter+1 +1, xblock_counter-1 +1) = nen_array(yblock_counter+1 +1, xblock_counter-1 +1) - 1;
                    end
                    if(yblock_counter<blocks_per_column-1)
                        nen_array(yblock_counter+1 +1, xblock_counter +1) = nen_array(yblock_counter+1 +1, xblock_counter +1) - 1;
                    end
                    if(yblock_counter<blocks_per_column-1 && xblock_counter<blocks_per_line-1)
                        nen_array(yblock_counter+1 +1, xblock_counter+1 +1) = nen_array(yblock_counter+1 +1, xblock_counter+1 +1) - 1;
                    end
                end
            end


            % debugging -------------------
            if (debug == 1 && block_size == block_size_min)
                figure('Name', 'current reconstruction');
                cur_rec = reconstructed_img;
                imshow(double(cur_rec/255));
                drawnow;
                refresh;
            end
            %------------------------------

            % set parameters for next extrapolation tasks (higher texture)
            block_size = block_size/2;
            border_width = (fft_size-block_size)/2;
            if block_size == 8
                threshold_stddev = threshold_stddev_LUT(2);
                rho = fsr_parameters.rhos(2);
            end
            if block_size == 4
                threshold_stddev = threshold_stddev_LUT(3);
                rho = fsr_parameters.rhos(3);
            end
            if block_size == 2
                rho = fsr_parameters.rhos(4);
            end

            % terminate function - no heterogeneous blocks left
            if isempty(set_later)
                break;
            end
        end
    else
        error('Invalid processing order selected!');
    end

end
