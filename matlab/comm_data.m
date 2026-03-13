clear all;
clc;

%% System parameters
lambda = 1;
Nt = 64;   % number of transmit antennas
M  = 4;    % number of users (Nr=1 each)
L  = 15;   % number of rays per user

k_wave = 2*pi/lambda;
d      = lambda/2;
n      = (0:Nt-1)';   % transmit antenna indices

n_train = 500;
n_test  = 100;
K       = 1;  % single subcarrier (no frequency dependence)

%% Generate training channels  (K x n_train x M x Nt)
H_train = zeros(K, n_train, M, Nt);
for ii = 1:n_train
    for m = 1:M
        h = zeros(1, Nt);
        for l = 1:L
            alpha = (randn + 1i*randn) / sqrt(2);
            phi   = 2*pi*rand;
            a_t   = (1/sqrt(Nt)) * exp(1i * k_wave * d * n * sin(phi));
            h     = h + sqrt(Nt/L) * alpha * a_t.';
        end
        H_train(1, ii, m, :) = h;
    end
end

%% Generate test channels  (K x n_test x M x Nt)
H_test = zeros(K, n_test, M, Nt);
for ii = 1:n_test
    for m = 1:M
        h = zeros(1, Nt);
        for l = 1:L
            alpha = (randn + 1i*randn) / sqrt(2);
            phi   = 2*pi*rand;
            a_t   = (1/sqrt(Nt)) * exp(1i * k_wave * d * n * sin(phi));
            h     = h + sqrt(Nt/L) * alpha * a_t.';
        end
        H_test(1, ii, m, :) = h;
    end
end

%% Save
save('comm_data.mat', 'H_train', 'H_test');