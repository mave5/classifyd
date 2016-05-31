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
disp('caffe mode gpu 2')


% load model and weights from dir that you trained the model
model = '../caffe/net_deploy.prototxt';
weights = '../caffe/trainedmodels/net_iter_12000.caffemodel';
%weights = '../caffe/score_0p76/weights/net_iter_70000.caffemodel';
disp(['loading weights   ', weights(end-24:end)]);
% create net
net = caffe.Net(model, weights, 'test'); % create net and load weights
disp('net was created !')


%% test data
disp('please wait to load test data')
pathname=('/data/dlnets/kaggle/imgs/test/');
listofjpgs=dir([pathname,'/*.jpg']);

% forwar to network
N=46;%size(I,4);
%classes=zeros(N,1);
net_out=zeros(N,10);
meanRGB=[80.25,97.05,95.4]*1;

testData=zeros(240,320,'uint8');
cntx=size(testData,1)/2;
cnty=size(testData,2)/2;
cropsize=224;
cx=cntx-cropsize/2:cntx+cropsize/2-1;
cy=cnty-cropsize/2:cnty+cropsize/2-1;

disp('please wait ...')
for k=1:N%numel(listofjpgs)
    img=fullfile(pathname,listofjpgs(k).name);
    
    testData=imresize(imread(img),0.5);
    
    testDataS(:,:,1)=single(testData(:,:,1,:))-meanRGB(1);
    testDataS(:,:,2)=single(testData(:,:,2,:))-meanRGB(2);
    testDataS(:,:,3)=single(testData(:,:,3,:))-meanRGB(3);

    %disp('please wait data is fed to net ...')
    res = net.forward({single(testDataS(cx,cy,:))});
    %net_out(k,:) = softmax(res{1},1);
    net_out(k,:) = res{1};
        
   
end
    scores=net_out;
    

  
[~,ind]=max(scores,[],2);
ind=ind-1;


% load sample test labels
filename='./results/sampletest.csv';
T=readtable(filename);

% numbers to compare
N2=46;

% images file name and labels
img2=T.img;
Labels=T.Label;

% compare
accu_ind=find(Labels(1:N2,1)==ind(1:N2,1));
accuracy=numel(accu_ind)/N2
error_ind=setdiff([1:N2],accu_ind);
error=numel(error_ind)/N2

% logloss
loss=0;
for k2=1:N2
    pij=scores(k2,Labels(k2)+1);
    pij=max(min(pij,1-1e-15),1e-15);
    loss=loss+log(pij);
end
logloss=-loss/N2;
disp(['logloss = ', num2str(logloss)])



%%

