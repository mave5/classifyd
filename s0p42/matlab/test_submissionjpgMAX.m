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
weights = '../caffe/trainedmodels/net_iter_36000.caffemodel';
disp(['loading weights   ', weights(end-24:end)]);
% create net
net = caffe.Net(model, weights, 'test'); % create net and load weights
disp('net was created !')


%% test data
disp('please wait to load test data')
pathname=('/data/dlnets/kaggle/imgs/test/');
filelist=dir([pathname,'/*.jpg']);

N=numel(filelist);
scores=zeros(N,10);
meanRGB=[80.25,97.05,95.4]*1;

testData=zeros(240,320,'uint8');


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

M=5;

disp('please wait ...')
for k1=1:numel(filelist)
    if rem(k1,100)==0
        k1
    end
    img=fullfile(pathname,filelist(k1).name);

    testData=imresize(imread(img),0.5);
    net_outC=zeros(M,10);
    for k2=1:M
        %rndx=randi(size(testData,1)-cropsize-1);
        %rndy=randi(size(testData,2)-cropsize-1);
        %cx=rndx:rndx+cropsize-1;
        %cy=rndy:rndy+cropsize-1;

        testDataS(:,:,1)=single(testData(:,:,1))-meanRGB(1);
        testDataS(:,:,2)=single(testData(:,:,2))-meanRGB(2);
        testDataS(:,:,3)=single(testData(:,:,3))-meanRGB(3);

        testDataC=testDataS(cx(k2,:),cy(k2,:),:);
        res = net.forward({testDataC});
        %net_outC(k2,:) = softmax(res{1},1);
        net_outC(k2,:) = res{1};
    end
    scores(k1,:)=max(net_outC);
        
   
end


%% save as csv file
img={filelist.name}';
c0=scores(:,1);
c1=scores(:,2);
c2=scores(:,3);
c3=scores(:,4);
c4=scores(:,5);
c5=scores(:,6);
c6=scores(:,7);
c7=scores(:,8);
c8=scores(:,9);
c9=scores(:,10);
%tmp={imgnm{1:N}}';
submission=table(img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9);
disp(submission(1:5,:))
writetable(submission,'./results/submission.csv')
writetable(submission,'/home/exx/Dropbox/kaggle/submission.csv')
