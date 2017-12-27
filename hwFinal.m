%???????????????? ????????,???????????????
% Why doesn't MATLAB save Japanese characters
% HOW AM I SUPPOSED TO PUT WEEB QUOTES IN MY CODE
% IF MATLAB DOESN'T SAVE JAPANESE CHARACTERS ?!
% WHO DID THIS?!?!?!?!?


%find current main folder
maindir = fullfile(pwd,'final_project');
%load the masked regions we created for the database
load(fullfile(maindir,'RegionsOfInterest.mat'));

%main dir of reference images
imageBagMainDir = fullfile(maindir,'DataBase');
%main dir of resized images (they're all 100x100)
imageBagResizedDir = fullfile(maindir,'DB_Resized');
height = 100;
width = 100;

%Used to resize all ref. images. One-time deal, no need to do it more than
%once.
%massResizeImages(maindir, imageBagMainDir, '.jpg', height, width);

%find all resized image names in the directory.
newImagefiles = dir(fullfile(imageBagResizedDir, '*.jpg'));
%all file names listed in a cell
filenames = {newImagefiles.name};
%total amount of images (100, but we're only using the first 50)
filecount = length(filenames);

%5 elements from each subgroup

% Format of each column
% redavg = 0;
% greavg = 0;
% bluavg = 0;
% Lavg = 0;
% aavg = 0;
% bavg = 0;
% initialise empty matrix
colorAvgData = zeros(6, (filecount/10));

% divided by 2 because I haven't found a use-case for the non-banana images
% There's one subset for every 5 images
for i= 1:(filecount/10)
    %initialise values above for each file entry
    r_a = 0;
    g_a = 0;
    b_a = 0;
    L_a = 0;
    a_a = 0;
    bb_a = 0;
    for j=1:(filecount/20)
        index = (i-1)*(filecount/20) + j;
        img = imread(fullfile(imageBagResizedDir,filenames{index}));
        %img_lab = rgb2lab(img);

        %regioncells were hand-selected
        img_mask = regioncells{index};

        %Get only the R-G-B values...
        img_r = img(:,:,1);
        img_g = img(:,:,2);
        img_b = img(:,:,3);

        img_mask = cast(img_mask, class(img_r));

        %...of the regions we selected for them.
        img_mask_r = img_mask .* img_r;
        img_mask_g = img_mask .* img_g;
        img_mask_b = img_mask .* img_b;

        img_mask_r = cast(img_mask_r, class(r_a));
        img_mask_g = cast(img_mask_g, class(r_a));
        img_mask_b = cast(img_mask_b, class(r_a));

        %purely used to find the number of non-black pixels
        threshold_r = (img_mask_r > 0);
        threshold_g = (img_mask_g > 0);
        threshold_b = (img_mask_b > 0);   
        rcount = sum(sum(threshold_r));
        gcount = sum(sum(threshold_g));
        bcount = sum(sum(threshold_b));

        %The masked images hold the intensity values.
        %Find the average by dividing their sum by the pixel total.
        r_a = r_a + sum(sum(img_mask_r)) / rcount;
        g_a = g_a + sum(sum(img_mask_g)) / rcount;
        b_a = b_a + sum(sum(img_mask_b)) / rcount;
        
        %UNUSED - L*a*b-space based filtering
%         cform = makecform('srgb2lab');
%         lab_img = applycform(masked_full,cform);
%         
%         ab = double(lab_img(:,:,2:3));
%         nrows = size(ab,1);
%         ncols = size(ab,2);
%         ab = reshape(ab,nrows*ncols,2);
%         
%         nColors = 3;
%         [cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
%             'Replicates',3);
%         
%         pixel_labels = reshape(cluster_idx,nrows,ncols);
%         
%         segmented_images = cell(1,3);
%         rgb_label = repmat(pixel_labels,[1 1 3]);
%         imshow(pixel_labels,[]), title('image labeled by cluster index');
%         
%         for k = 1:nColors
%             color = masked_full;
%             color(rgb_label ~= k) = 0;
%             segmented_images{k} = color;
%         end

    end
    colorAvgData(:,i) = [r_a;g_a;b_a;L_a;a_a;bb_a];
end

%10 subgroups' worth of data
colorAvgData = colorAvgData / (filecount/20);

%Threshold base for our color values.
%This actually changes for each colour.
threshold = int16(40);

%+++++++++++READ TEST INPUT HERE+++++++++++
sampleimg = imread(fullfile(maindir,'testinput7.jpg'));
[h2,w2,~] = size(sampleimg);

samplemask = uint8(zeros(h2,w2));
%Compare the input to EVERY subgroup data we have
%spoilers, we have 10
for index=1:(filecount/10)
    for i=1:h2
        for j=1:w2
            l_r = double(sampleimg(i,j,1));
            l_g = double(sampleimg(i,j,2));
            l_b = double(sampleimg(i,j,3));
            %Threshold percentages are based on the intensity
            %of the colour in the database averages
            check = (abs(l_r - colorAvgData(1,index)) <= threshold/2) & ...
                (abs(l_g - colorAvgData(2,index)) <= threshold*5/8) & ...
                (abs(l_b - colorAvgData(3,index)) <= threshold/10);
            if (check)
                samplemask(i,j) = 1;
            end
        end
    end
end

%Mask the image with the "valid" bits so only the relevant parts are shown
%In an utopic code, the "relevant" parts would strictly be the banana.
%This one is just acceptable.
masked_r = sampleimg(:,:,1) .* samplemask;
masked_g = sampleimg(:,:,2) .* samplemask;
masked_b = sampleimg(:,:,3) .* samplemask;

%Merge them into an RGB image.
masked_full = cat(3,masked_r,masked_g,masked_b);

graymask = rgb2gray(masked_full);
%filter grayscale based on intensity
binaryimg = graymask > 128;
binaryimg = imfill(binaryimg,'holes');
%Show final B&W image
imshow(binaryimg,[]);


%--------Was used to manually crop regions from reference images--------
% Also prints out irrelevant garbage like the color histogram of the
% cropped areas.
% histr = uint16(zeros(height,width,1));
% histg = uint16(zeros(height,width,1));
% histb = uint16(zeros(height,width,1));
% 
% regioncells = cell(floor(filecount/2),1);
% 
% for i=1:50
%     imgtest = imread(fullfile(imageBagResizedDir,filenames{i}));
%     BW = roipoly(imgtest)
%     regioncells{i} = BW;
%     
%         
%     imgr = imgtest(:,:,1);
%     imgg = imgtest(:,:,2);
%     imgb = imgtest(:,:,3);
%     
%     BW = cast(BW,class(imgr));
%     
%     maskedImageR = imgr .* BW;
%     maskedImageG = imgg .* BW;
%     maskedImageB = imgb .* BW;
%     
%     maskedImage = cat(3,maskedImageR,maskedImageG,maskedImageB);
%     
%     imshow(maskedImage)
%     
%     waitforbuttonpress;


%     [yRed, x] = imhist(imgr);
%     [yGreen, x] = imhist(imgg);
%     [yBlue, x] = imhist(imgb);

    %plot(x, yRed, 'Red', x, yGreen, 'Green', x, yBlue, 'Blue');
% end

% save('RegionsOfInterest.mat', regioncells);

% for i=1:(filecount/2)
%     img = imread(fullfile(imageBagResizedDir,filenames{i}));
%     img(:,:,1);
%     
%     histr = histr + uint16(img(:,:,1));
%     histg = histg + uint16(img(:,:,2));
%     histb = histb + uint16(img(:,:,3));
% end
% 
% close all;
% 
% histr = double(uint8(histr ./ 50));
% hist(histr);
% title('Red Values Dist.');
% histg = double(uint8(histg ./ 50));
% figure;
% hist(histg);
% title('Green Values Dist.');
% histb = double(uint8(histb ./ 50));
% figure;
% hist(histb);
% title('Blue Values Dist.');

%montage(imageBagSet.Files(1:2:numFiles),'Size', [5 5]);


%Resize the database files to a reasonable size
function massResizeImages(maindir, filedir, extension, height, width)
    imagefiles = dir(fullfile(filedir,strcat('*',extension)));
    filenames = {imagefiles.name};
     for i=1:length(filenames)
        img = imread(fullfile(filedir,filenames{i}));
        img_save = imresize(img, [height width]);
        %save it under the "exImage' prefix, WITH A ZERO AT THE START OF
        %THE NUMBER
        name_save = strcat('exImage', sprintf('%03d',i), extension);
        savefolder = fullfile(maindir,'DB_Resized');
        if ~exist(savefolder, 'dir')
            mkdir(savefolder);
        end
        imwrite(img_save, fullfile(savefolder, name_save));
     end
end

%????????????????????????????????, ?? ?????????? ?? ?????!??