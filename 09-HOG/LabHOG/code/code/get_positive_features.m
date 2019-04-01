% Starter code prepared by James Hays for CS 143, Brown University
% This function should return all positive training examples (faces) from
% 36x36 images in 'train_path_pos'. Each face should be converted into a
% HoG template according to 'feature_params'. For improved performance, try
% mirroring or warping the positive training examples.

function features_pos = get_positive_features(train_path_pos, feature_params)
% 'train_path_pos' is a string. This directory contains 36x36 images of
%   faces
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.


% 'features_pos' is N by D matrix where N is the number of faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray
%'data/caltech_faces/Caltech_CropFaces ';

train_path_pos='data/caltech_faces/Caltech_CropFaces ';
image_files = dir( fullfile( 'data','caltech_faces','Caltech_CropFaces', '*.jpg') ); %Caltech Faces stored as .jpg
num_images = length(image_files);
dim=(feature_params.template_size / feature_params.hog_cell_size)^2 * 31;
im1= imread(fullfile( 'data','caltech_faces','Caltech_CropFaces',image_files(1).name));
images.Images=im1;
cellSize = feature_params(1).hog_cell_size;

features_pos.HOG =vl_hog(im2single(im1), cellSize);
for i=2:num_images
    im2=imread(fullfile( 'data','caltech_faces','Caltech_CropFaces',image_files(i).name));
    HOG2 = vl_hog(im2single(im2), cellSize);
    class(HOG2)
    images(i).Images=im2;
    features_pos(i,:)=reshape(HOG2,1,dim);    
end


% placeholder to be deleted - THIS ONLY WORKS FOR THE INITIAL DEMO
%features_pos = rand(100, (feature_params.template_size / feature_params.hog_cell_size)^2 * 31);

