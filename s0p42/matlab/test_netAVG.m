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
meanRGB=[80.25,97.05,95.4];
%meanBGR=[95.4,97.05,80.25];
testDataS(:,:,1,:)=single(testData(:,:,1,:))-meanRGB(1);
testDataS(:,:,2,:)=single(testData(:,:,2,:))-meanRGB(2);
testDataS(:,:,3,:)=single(testData(:,:,3,:))-meanRGB(3);




%% load model and weights from dir that you trained the model
model = '../caffe/net_deploy.prototxt';
weights = '../caffe/trainedmodels/net_iter_36000.caffemodel';
disp(['loading weights   ', weights(end-24:end)]);

% create net
net = caffe.Net(model, weights, 'test'); % create net and load weights

 disp('please wait data is fed to net ...')

 % forwar to network
N=size(testData,4);
classes=zeros(N,1);
M=5;
cropsize=224;

Wx=size(testData,1);
Wy=size(testData,2);

% center coordinate
cntx=Wx/2;
cnty=Wy/2;

% crop coordinate from center
cx(1,:)=cntx-cropsize/2:cntx+cropsize/2-1;
cy(1,:)=cnty-cropsize/2:cnty+cropsize/2-1;

% crop coordinate from top left
cx(2,:)=1:cropsize;
cy(2,:)=1:cropsize;

% crop coordinate from top right
cx(3,:)=1:cropsize;
cy(3,:)=Wy-cropsize+1:Wy;

% crop coordinate from bottom left
cx(4,:)=Wx-cropsize+1:Wx;
cy(4,:)=1:cropsize;

% crop coordinate from bottom right
cx(5,:)=Wx-cropsize+1:Wx;
cy(5,:)=Wy-cropsize+1:Wy;



net_out=zeros(N,10);
for k1=1:N %size(testData,4)
    disp(['testing image #', num2str(k1)]);
    for k2=1:M
        %rndx=randi(size(testData,1)-cropsize-1);
        %rndy=randi(size(testData,2)-cropsize-1);
        %cx=rndx:rndx+cropsize-1;
        %cy=rndy:rndy+cropsize-1;

        testDataC=testDataS(cx(k2,:),cy(k2,:),:,k1);
        res = net.forward({testDataC});
        %net_outC(k2,:) = softmax(res{1},1);
        net_outC(k2,:) = res{1};
    end
    net_out(k1,:)=max(net_outC);
    % convert to logical values
    [~,classes(k1)]=max(net_out(k1,:));    
    %if classes(k1)~=testLabels(1,k1)+1
        %net_outC
        %classes(k1)-1
        %testLabels(1,k1)
        %pause
    %end
end

% 
classes=classes-1;

% calculate accuracy
labels=squeeze(testLabels);
labels=reshape(labels,N,1);
accu_ind=find(classes(1:N)==labels(1:N));
accu=numel(accu_ind)/N*100;
disp(['Accuracy= ', num2str(accu)])


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




