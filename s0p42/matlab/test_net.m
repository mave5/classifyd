% load caffe model and trained weights into MATLAB 
% test the net using test data and visualize
clear
clc
close all
addpath functions
%%

% find the path to caffe on your computer and and set it here to make life easy
caffe_root='/usr/local/caffeapril2016/';

% Add caffe/matlab to your Matlab search PATH to use matcaffe
if exist([caffe_root,'matlab/+caffe'], 'dir')
  addpath([caffe_root,'matlab']);
else
  accu('path does not exist');
end

% set cafe mode
caffe.set_mode_gpu();
caffe.set_device(2); % GPU ID

%% test data
disp('please wait to load test data')
load('/data/dlnets/kaggle/matfiles/testdata_lmdb.mat');
%load('/data/kaggle/matfiles/traindata_lmdb.mat');
%testData=trainData;
%testLabels=trainLabels;
%load('../matfiles/meanstd.mat')

% subtract mean value
meanRGB=[80.25,97.05,95.4]*1;

testDataS(:,:,1,:)=single(testData(:,:,1,:))-meanRGB(1);
testDataS(:,:,2,:)=single(testData(:,:,2,:))-meanRGB(2);
testDataS(:,:,3,:)=single(testData(:,:,3,:))-meanRGB(3);

%crop from center
cntx=size(testData,1)/2;
cnty=size(testData,2)/2;
cropsize=224;
cx=cntx-cropsize/2:cntx+cropsize/2-1;
cy=cnty-cropsize/2:cnty+cropsize/2-1;
testDataC=testDataS(cx,cy,:,:);
%testDataC=testDataS;

%% load model and weights from dir that you trained the model
model = '../caffe/net_deploy.prototxt';
weights = '../caffe/trainedmodels/net_iter_36000.caffemodel';
disp(['loading weights   ', weights(end-24:end)]);

% create net
net = caffe.Net(model, weights, 'test'); % create net and load weights

 disp('please wait data is fed to net ...')

 % forwar to network
N=size(testDataC,4);
classes=zeros(N,1);
net_out=zeros(N,10);
for k1=1:N %size(testData,4)
    %disp(['testing image #', num2str(k1)]);
    
    res = net.forward({testDataC(:,:,:,k1)});
    %net_out(k1,:) = softmax(res{1},1);
    net_out(k1,:) = res{1};

    % convert to logical values
    [~,classes(k1)]=max(net_out(k1,:)); 
end

% 
classes=classes-1;

% calculate accuracy
labels=squeeze(testLabels);
labels=reshape(labels,N,1);
accu_ind=find(classes(1:N)==labels(1:N));
accu=numel(accu_ind)/N*100;
disp(['Accuracy= ', num2str(accu)])

% error
erro_ind=find(classes(1:N)~=labels(1:N));
k1=erro_ind(4);
imshow(testDataC(:,:,:,k1));title(['predicted: ',num2str(classes(k1)), 'label: ', num2str(testLabels(1,k1))])
net_out(k1,:);


% logloss
loss=0;
for k2=1:N
    pij=net_out(k2,labels(k2)+1);
    pij=max(min(pij,1-1e-15),1e-15);
    loss=loss+log(pij);
end
logloss=-loss/N;
disp(['logloss = ', num2str(logloss)])
%%
n1=randi(5000);
imshow(testDataC(:,:,:,n1));
title(['predicted: ',num2str(classes(n1)), '   Label: ', num2str(labels(n1)), '   frame #', num2str(n1)])




