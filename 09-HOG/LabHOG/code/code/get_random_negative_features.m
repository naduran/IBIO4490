% Starter code prepared by James Hays for CS 143, Brown University
% This function should return negative training examples (non-faces) from
% any images in 'non_face_scn_path'. Images should be converted to
% grayscale, because the positive training data is only available in
% grayscale. For best performance, you should sample random negative
% examples at multiple scales.

function features_neg = get_random_negative_features(non_face_scn_path, feature_params, num_samples)
% 'non_face_scn_path' is a string. This directory contains many images
%   which have no faces in them.
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.
% 'num_samples' is the number of random negatives to be mined, it's not
%   important for the function to find exactly 'num_samples' non-face
%   features, e.g. you might try to sample some number from each image, but
%   some images might be too small to find enough.

% 'features_neg' is N by D matrix where N is the number of non-faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray

non_face_scn_path='data/train_non_faces_scenes';
image_files=dir( fullfile( 'data','train_non_face_scenes', '*.jpg') ); %Caltech Faces stored as .jpg  
num_images = length(image_files);

%cd data/train_non_face_scenes/
im1= imread( fullfile( 'data','train_non_face_scenes',image_files(1).name));
images.Images=im1;
cellSize = feature_params(1).hog_cell_size;
features_neg.HOG =vl_hog(im2single(im1), cellSize);
dim=(feature_params.template_size / feature_params.hog_cell_size)^2 * 31;
if num_samples>num_images
    error('The number of samples its too big');
end
for i=2:num_samples
    im2=imread(fullfile( 'data','train_non_face_scenes',image_files(i).name));
    HOG2 = vl_hog(im2single(im2), cellSize);
    images(i).Images=im2;
    features_neg(i,:)=reshape(HOG2,1,dim);    
end
% placeholder to be deleted - THIS ONLY WORKS FOR THE INITIAL DEMO
%features_neg = rand(100, (feature_params.template_size / feature_params.hog_cell_size)^2 * 31);
