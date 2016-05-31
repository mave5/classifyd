% convert images into two datasets for train and validation
clc
clear
close all
%%

% load training images
path2imgs='/data/kaggle/imgs/train';

% load drivers list for train
filename='/data/kaggle/kaggledata/driverTrain_imgs_list.csv';
T_train=readtable(filename);

% extract subjeects, imgs, classes
sbjs_train=T_train.sbjs_train;
class_train=T_train.class_train;
imgs_train=T_train.imgs_train;


% initialize 
Ntrain=numel(sbjs_train);
Itrain=zeros(480,640,3,Ntrain,'uint8');
Ltrain=zeros(1,Ntrain,'uint8');

disp('please wait while loading images')
for k1=1:Ntrain
    disp(['loading ',num2str(k1)])
    fnm=fullfile(path2imgs,class_train{k1},imgs_train{k1});
    Itrain(:,:,:,k1)=imread(fnm);
    Ltrain(k1)=str2double(class_train{k1}(end));
end

% shuffle images
train_ind=randperm(Ntrain);
trainData=Itrain(:,:,:,train_ind);
trainLabels=Ltrain(1,train_ind);
disp('train data shuffled ')

% downsample images
imr1=[224 224];
trainData=imresize(trainData,imr1);

% save into mat file
disp('please wait to save train data into mat file ...')
pathname2='../matfiles';
trainfnlmdb=fullfile(pathname2,'traindata_lmdb.mat');
save(trainfnlmdb,'trainLabels','trainData','-v7.3');
disp(' mat files was saved for lmdb processing !');
save([pathname2,'/trainind.mat'],'train_ind')
%%

% load drivers list for train
filename='/data/kaggle/kaggledata/driverVal_imgs_list.csv';
Tval=readtable(filename);

% extract subjeects, imgs, classes
sbjs_val=Tval.sbjs_val;
class_val=Tval.class_val;
imgs_val=Tval.imgs_val;


% initialize 
Nval=numel(sbjs_val);
Ival=zeros(480,640,3,Nval,'uint8');
Lval=zeros(1,Nval,'uint8');

disp('please wait while loading images')
for k1=1:Nval
    disp(['loading ',num2str(k1)])
    fnm=fullfile(path2imgs,class_val{k1},imgs_val{k1});
    Ival(:,:,:,k1)=imread(fnm);
    Lval(k1)=str2double(class_val{k1}(end));
end

% shuffle images
test_ind=randperm(Nval);
testData=Ival(:,:,:,test_ind);
testLabels=Lval(1,test_ind);
disp('test data shuffled ')

% downsample test data
testData=imresize(testData,imr1);

disp('please wait to save test data into mat file ...')
% save into mat file
pathname2='../matfiles';
trainfnlmdb=fullfile(pathname2,'testdata_lmdb.mat');
save(trainfnlmdb,'testLabels','testData','-v7.3');
disp(' test mat files was saved for lmdb processing !');
save([pathname2,'/testind.mat'],'test_ind')



%% sample image
n1=randi(1453);
imshow(testData(:,:,:,n1))
title(num2str(testLabels(n1)))
