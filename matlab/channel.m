Q = 15; % number of scatters
N_t = 64; % Number of transmit antennas
K = 4;
batch = 5000;
H = zeros(1, batch, K, N_t);
for b = 1: batch
    for k = 1:K
        % Complex Gaussian path gains: CN(0,1)
        alpha_k = (randn(Q,1) + 1i*randn(Q,1)) / sqrt(2);

        % Single-line solution
        AoD = exp(1i * pi * sin(2 * pi * rand(1, Q)) .* (0:N_t-1)') / sqrt(N_t); % N_t x Q

        % Channel vector h_k = sum_q alpha_q * a(phi_q)
        h_k = AoD * alpha_k; % Nt x 1
        H(1, b, k, :) = h_k.';
    end
end

filename = 'H_train.mat';
save(filename, 'H');
