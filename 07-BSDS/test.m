%Rename segm-test cells
segm_t=dir('segm-test');
chdir('segm-test');
for i=3:length(segm_t)
    im=segm_t(i).name;
    im=im(1:find(im=='.'));
    im=[im 'mat'];
    %movefile(segm_t(i).name,im);
end
%% Copy images to new diretory adapted on segm-test
chdir('..')
mkdir('images_test')
addpath('images_test')
mkdir('ground_test')
addpath('ground_test')
segm_t=dir('segm-test');
imgs=dir('BSR/BSDS500/data/images/test');
for i=3:length(segm_t)
    for j=1:length(imgs)
    im=segm_t(i).name;
    im_t=imgs(j).name;
    im=im(1:find(im=='.'));
    im_t=im_t(1:find(im_t=='.'));
    if length(im)==length(im_t)
    if im_t==im
    orig=['BSR/BSDS500/data/images/test/' im_t 'jpg'];
    dest=['images_test/' im_t 'jpg'];
    orig_g=['BSR/BSDS500/data/groundTruth/test/' im_t 'mat'];
    dest_g=['ground_test/' im_t 'mat'];
    copyfile(orig,dest)
    copyfile(orig_g,dest_g)
    end
    end
    end
end

%% Convert matrix to gray scale
segm_t=dir('segm-test');
chdir('segm-test');
for i=3:length(segm_t)
 a=load(segm_t(i).name);
 segs=a.segm;
 for j=1:length(segs)
     segs{j}=rgb2gray(segs{j});
     segs{j}=segs{j}+1;
 end
 save(segm_t(i).name,'segs')
end
chdir('..')
%%
imgDir = 'images_test';
gtDir = 'ground_test';
inDir = 'segm-test';
outDir = 'eval/test_all_segms';
mkdir(outDir);
nthresh = 99;
addpath('BSR/bench_fast/benchmarks')
tic;
allBench_fast(imgDir, gtDir, inDir, outDir, nthresh);
toc;

plot_eval(outDir)