%% Initiate the F and W

%% Load H from dataset/64TX_4UE_4RF/train_data_matlab.mat
load('../dataset/64TX_4UE_4RF/train_data_matlab.mat');
disp('Loaded H from train_data_matlab.mat');
disp('Size of H_train:');
disp(size(H_train));


H_test_t = squeeze(H_train(1,1,:,:)); % Use the first sample for testing
disp('Size of H_test:');
disp(H_test);

H_test = H_test_t';

%% Initialize F

G = H_test; %%since K = M
phi = angle(G);
F_0 = exp(phi);
disp(F_0)