% load caffe model and trained weights into MATLAB 
% test the net using test data and visualize
clear
clc
close all
addpath functions
%%

% find the path to caffe on your computer and and set it here to make life easy
caffe_root='/usr/local/caffe/';

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
cntx=size(testData,1)/2;
cnty=size(testData,2)/2;
cropsize=224;
cx=cntx-cropsize/2:cntx+cropsize/2-1;
cy=cnty-cropsize/2:cnty+cropsize/2-1;

disp('please wait ...')
for k=1:numel(filelist)
    if rem(k,100)==0
        k
    end
    img=fullfile(pathname,filelist(k).name);

    testData=imresize(imread(img),0.5);
    M=10;net_outC=zeros(M,10);
    for k2=1:M
        rndx=randi(size(testData,1)-cropsize-1);
        rndy=randi(size(testData,2)-cropsize-1);
        cx=rndx:rndx+cropsize-1;
        cy=rndy:rndy+cropsize-1;

        testDataS(:,:,1)=single(testData(:,:,1,:))-meanRGB(1);
        testDataS(:,:,2)=single(testData(:,:,2,:))-meanRGB(2);
        testDataS(:,:,3)=single(testData(:,:,3,:))-meanRGB(3);

        res = net.forward({single(testDataS(cx,cy,:))});
        %net_outC(k2,:) = softmax(res{1},1);
        net_outC(k2,:) = res{1};
    end
    scores(k,:)=mean(net_outC);
        
   
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
