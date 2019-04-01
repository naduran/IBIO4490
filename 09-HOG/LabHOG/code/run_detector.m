function [bboxes, confidences, image_ids] = run_detector(test_scn_path, w, b, feature_params)
% 'test_scn_path' is a string. This directory contains images which may or
%    may not have faces in them. This function should work for the MIT+CMU
%    test set but also for any other images (e.g. class photos)
% 'w' and 'b' are the linear classifier parameters
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.

% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
%   [x_min, y_min, x_max, y_max] for detection i.
%   Remember 'y' is dimension 1 in Matlab!
% 'confidences' is Nx1. confidences(i) is the real valued confidence of
%   detection i.
% 'image_ids' is an Nx1 cell array. image_ids{i} is the image file name
%   for detection i. (not the full path, just 'albert.jpg')

% The placeholder version of this code will return random bounding boxes in
% each test image. It will even do non-maximum suppression on the random
% bounding boxes to give you an example of how to call the function.

% Your actual code should convert each test image to HoG feature space with
% a _single_ call to vl_hog for each scale. Then step over the HoG cells,
% taking groups of cells that are the same size as your learned template,
% and classifying them. If the classification is above some confidence,
% keep the detection and then pass all the detections for an image to
% non-maximum suppression. For your initial debugging, you can operate only
% at a single scale and you can skip calling non-maximum suppression.
test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));

bboxes = zeros(0,4);
confidences = zeros(0,1);


image_ids = cell(0,1);

s_win = feature_params.template_siz/ feature_params.hog_cell_size;

for i = 1:length(test_scenes)
    
    cur_bboxes = zeros(0, 4);
    cur_confidences = zeros(0);
    cur_image_ids = {};
      
    
    fprintf('Detecting faces in %s\n', test_scenes(i).name)
    img = imread( fullfile( test_scn_path, test_scenes(i).name ));
    img = single(img)/255;
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    
 
    scales = [1,0.8,0.6,0.4,0.2]; 
    
    for s = scales
        img= imresize(img, s);
        HOG = vl_hog(img, feature_params.hog_cell_size); 
        for i = 1:size(HOG,1) - s_win + 1
            for j = 1:size(HOG,2) - s_win + 1
                
                cut = hog(i + s_win - 1, j + s_win - 1, :);
                cut= reshape(cut,1, s_win^2 * 31);
                confidence = cut * w + b; 
                
                if confidence > th
                   min_x = ((j - 1) * feature_params.hog_cell_size + 1) / s;
                   min_y = ((i - 1) * feature_params.hog_cell_size + 1) / s;
                   max_x = ((j + s_win - 1) * feature_params.hog_cell_size) / s;
                   max_y = ((i + s_win - 1) * feature_params.hog_cell_size) / s;
                    
                    cur_bboxes = [cur_bboxes; [min_x, min_y, max_x, max_y]];
                    cur_confidences = [cur_confidences; confidence];
                end
            end
        end
    end
    
    patches = size(cur_bboxes, 1);
    cur_image_ids(1:patches,1) = {test_scenes(i).name};
    
    [is_maximum] = non_max_supr_bbox(cur_bboxes, cur_confidences, size(img));
    
    cur_confidences = cur_confidences(is_maximum,:);
    cur_bboxes      = cur_bboxes(     is_maximum,:);
    cur_image_ids   = cur_image_ids(  is_maximum,:);
    
    bboxes      = [bboxes;      cur_bboxes];
    confidences = [confidences; cur_confidences];
    image_ids   = [image_ids;   cur_image_ids];
    

end

References
    %Based on the method from https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj5/html/jkim844/index.html