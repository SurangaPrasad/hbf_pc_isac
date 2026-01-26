import numpy as np
import matplotlib.pyplot as plt
from utility import *
from PGA_models import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---- GPU Setup: Detect and configure device ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# ---- training and test the models ----

# Load training data
H_train, H_test0 = get_data_tensor(data_source)
H_test = H_test0[:, :test_size, :, :]
torch.manual_seed(3407)

# Move test data to GPU (training batches are transferred on-the-fly)
H_test = H_test.to(device)

# ====================================================== Conventional PGA ====================================
if run_conv_PGA == 1:
    # Object defining
    model_conv_PGA = PGA_Conv(step_size_conv_PGA).to(device)  # Move model to GPU

    # executing classical PGA on the test set
    R, at, theta, ideal_beam = get_radar_data(snr_dB, H_test)
    # Move radar data to GPU
    R = R.to(device)
    at = at.to(device)
    theta = torch.as_tensor(theta, device=device)
    ideal_beam = torch.as_tensor(ideal_beam, device=device)

    rate_iter_conv, beam_iter_conv, F_conv, W_conv = model_conv_PGA.execute_PGA(H_test, R, snr, n_iter_outer)
    # Keep results as torch tensors to avoid GPU↔CPU transfers
    rate_conv = torch.mean(rate_iter_conv, dim=0)
    beam_error_conv = torch.mean(beam_iter_conv, dim=0)
    iter_number_conv = torch.arange(n_iter_outer + 1, device=rate_conv.device, dtype=rate_conv.dtype)

# ====================================================== Unfolded PGA with J = 1 ====================================
if run_UPGA_J1 == 1:

    # Object defining
    model_UPGA_J1 = PGA_Conv(step_size_UPGA_J1).to(device)  # Move model to GPU
    # training procedure
    optimizer = torch.optim.Adam(model_UPGA_J1.parameters(), lr=learning_rate)
    train_losses, valid_losses = [], []

    for i_epoch in range(n_epoch):
        print(i_epoch)
        H_shuffeld = torch.transpose(H_train, 0, 1)[np.random.permutation(len(H_train[0]))]
        for i_batch in range(0, len(H_train), batch_size):
            H = torch.transpose(H_shuffeld[i_batch:i_batch + batch_size], 0, 1).to(device)
            snr_dB_train = np.random.choice(snr_dB_list)
            snr_train = 10 ** (snr_dB_train / 10)
            Rtrain, _, _, _ = get_radar_data(snr_dB_train, H)
            # Move radar data to GPU (H is already on GPU from H_train)
            Rtrain = Rtrain.to(device)
            
            __, __, F, W = model_UPGA_J1.execute_PGA(H, Rtrain, snr_train, n_iter_outer)
            loss = get_sum_loss(F, W, H, Rtrain, snr_train, batch_size)

            optimizer.zero_grad()  # zero the gradient buffers
            loss.backward()
            optimizer.step()  # Does the update

    # Save trained model
    torch.save(model_UPGA_J1.state_dict(), model_file_name_UPGA_J1)

    # Create new model and load states
    model_test = PGA_Conv(step_size_UPGA_J1).to(device)  # Move model to GPU
    model_test.load_state_dict(torch.load(model_file_name_UPGA_J1, map_location=device))  # Load to correct device

    # executing unfolded PGA on the test set
    Rtest, _, _, _ = get_radar_data(snr_dB, H_test)
    Rtest = Rtest.to(device)  # Move radar data to GPU
    
    rate_iter_UPGA_J1, beam_error_iter_UPGA_J1, F_UPGA_J1, W_UPGA_J1 = model_test.execute_PGA(H_test, Rtest, snr, n_iter_outer)
    rate_UPGA_J1 = torch.mean(rate_iter_UPGA_J1, dim=0)
    beam_error_UPGA_J1 = torch.mean(beam_error_iter_UPGA_J1, dim=0)
    iter_number_UPGA_J1 = torch.arange(n_iter_outer + 1, device=rate_UPGA_J1.device, dtype=rate_UPGA_J1.dtype)

# ============================================================= proposed unfolding PGA =================================
if run_UPGA_J20 == 1:

    # Object defining
    model_UPGA_J20 = PGA_Unfold_J20(step_size_UPGA_J20).to(device)  # Move model to GPU

    # training procedure
    optimizer = torch.optim.Adam(model_UPGA_J20.parameters(), lr=learning_rate)

    train_losses, valid_losses = [], []

    for i_epoch in range(n_epoch):
        print(i_epoch)
        H_shuffeld = torch.transpose(H_train, 0, 1)[np.random.permutation(len(H_train[0]))]
        for i_batch in range(0, len(H_train), batch_size):
            H = torch.transpose(H_shuffeld[i_batch:i_batch + batch_size], 0, 1).to(device)
            snr_dB_train = np.random.choice(snr_dB_list)
            snr_train = 10 ** (snr_dB_train / 10)
            Rtrain, _, _, _ = get_radar_data(snr_dB_train, H)
            # Move radar data to GPU (H is already on GPU from H_train)
            Rtrain = Rtrain.to(device)
            
            rate, __, F, W = model_UPGA_J20.execute_PGA(H, Rtrain, snr_train, n_iter_outer, n_iter_inner_J20)
            loss = get_sum_loss(F, W, H, Rtrain, snr_train, batch_size)
            # loss = -sum(sum(rate[1:]) / (K * batch_size))

            optimizer.zero_grad()  # zero the gradient buffers
            loss.backward()
            optimizer.step()  # Does the update

    # Save trained model
    torch.save(model_UPGA_J20.state_dict(), model_file_name_UPGA_J20)

    # test proposed model
    model_test = PGA_Unfold_J20(step_size_UPGA_J20).to(device)  # Move model to GPU
    model_test.load_state_dict(torch.load(model_file_name_UPGA_J20, map_location=device))  # Load to correct device
    
    Rtest, at, theta, ideal_beam = get_radar_data(snr_dB, H_test)
    # Move radar data to GPU
    Rtest = Rtest.to(device)
    at = at.to(device)
    theta = torch.as_tensor(theta, device=device)
    ideal_beam = torch.as_tensor(ideal_beam, device=device)
    
    rate_iter_UPGA_J20, beam_error_iter_UPGA_J20, F_UPGA_J20, W_UPGA_J20 = model_test.execute_PGA(H_test, Rtest, snr, n_iter_outer,
                                                                                             n_iter_inner_J20)
    rate_UPGA_J20 = torch.mean(rate_iter_UPGA_J20, dim=0)
    beam_error_UPGA_J20 = torch.mean(beam_error_iter_UPGA_J20, dim=0)
    iter_number_UPGA_J20 = torch.arange(n_iter_outer + 1, device=rate_UPGA_J20.device, dtype=rate_UPGA_J20.dtype)

# ============================================================= proposed unfolding PGA =================================
if run_UPGA_J10 == 1:

    # Object defining
    model_UPGA_J10 = PGA_Unfold_J10(step_size_UPGA_J10).to(device)  # Move model to GPU

    # training procedure
    optimizer = torch.optim.Adam(model_UPGA_J10.parameters(), lr=learning_rate)

    train_losses, valid_losses = [], []

    for i_epoch in range(n_epoch):
        print(i_epoch)
        H_shuffeld = torch.transpose(H_train, 0, 1)[np.random.permutation(len(H_train[0]))]
        for i_batch in range(0, len(H_train), batch_size):
            H = torch.transpose(H_shuffeld[i_batch:i_batch + batch_size], 0, 1).to(device)
            snr_dB_train = np.random.choice(snr_dB_list)
            snr_train = 10 ** (snr_dB_train / 10)
            Rtrain, _, _, _ = get_radar_data(snr_dB_train, H)
            # Move radar data to GPU (H is already on GPU from H_train)
            Rtrain = Rtrain.to(device)
            
            rate, __, F, W = model_UPGA_J10.execute_PGA(H, Rtrain, snr_train, n_iter_outer,
                                                        n_iter_inner_J10)
            loss = get_sum_loss(F, W, H, Rtrain, snr_train, batch_size)
            # loss = -sum(sum(rate[1:]) / (K * batch_size))

            optimizer.zero_grad()  # zero the gradient buffers
            loss.backward()
            optimizer.step()  # Does the update

    # Save trained model
    torch.save(model_UPGA_J10.state_dict(), model_file_name_UPGA_J10)

    # test proposed model
    model_test = PGA_Unfold_J10(step_size_UPGA_J10).to(device)  # Move model to GPU
    model_test.load_state_dict(torch.load(model_file_name_UPGA_J10, map_location=device))  # Load to correct device
    
    Rtest, at, theta, ideal_beam = get_radar_data(snr_dB, H_test)
    # Move radar data to GPU
    Rtest = Rtest.to(device)
    at = at.to(device)
    theta = torch.as_tensor(theta, device=device)
    ideal_beam = torch.as_tensor(ideal_beam, device=device)
    
    rate_iter_UPGA_J10, beam_error_iter_UPGA_J10, F_prop_UPGA_J10, W_prop_UPGA_J10 = model_test.execute_PGA(H_test, Rtest,
                                                                                                            snr,
                                                                                                            n_iter_outer,
                                                                                                            n_iter_inner_J10)
    rate_UPGA_J10 = torch.mean(rate_iter_UPGA_J10, dim=0)
    beam_error_UPGA_J10 = torch.mean(beam_error_iter_UPGA_J10, dim=0)
    iter_number_UPGA_J10 = torch.arange(n_iter_outer + 1, device=rate_UPGA_J10.device, dtype=rate_UPGA_J10.dtype)

# ============================================= Proposed unfolded PGA J=10 PC ============================================

if run_UPGA_J10_PC == 1:

    # Object defining
    model_UPGA_J10_PC = PGA_Unfold_J10_PC(step_size_UPGA_J10_PC).to(device)  # Move model to GPU

    # training procedure
    optimizer = torch.optim.Adam(model_UPGA_J10_PC.parameters(), lr=learning_rate)

    train_losses, valid_losses = [], []

    for i_epoch in range(n_epoch):
        print(i_epoch)
        H_shuffeld = torch.transpose(H_train, 0, 1)[np.random.permutation(len(H_train[0]))]
        for i_batch in range(0, len(H_train), batch_size):
            H = torch.transpose(H_shuffeld[i_batch:i_batch + batch_size], 0, 1).to(device)
            snr_dB_train = np.random.choice(snr_dB_list)
            snr_train = 10 ** (snr_dB_train / 10)
            Rtrain, _, _, _ = get_radar_data(snr_dB_train, H)
            # Move radar data to GPU (H is already on GPU from H_train)
            Rtrain = Rtrain.to(device)
            print(f'Batch {i_batch} processing')

            __ , __, F, W = model_UPGA_J10_PC.execute_PGA(H, Rtrain, snr_train, n_iter_outer, n_iter_inner_J10)
            loss = get_sum_loss(F, W, H, Rtrain, snr_train, batch_size)


            optimizer.zero_grad()  # zero the gradient buffers
            loss.backward()
            optimizer.step()  # Does the update 

    # Save trained model
    torch.save(model_UPGA_J10_PC.state_dict(), model_file_name_UPGA_J10_PC)

    # test proposed model
    model_test = PGA_Unfold_J10_PC(step_size_UPGA_J10_PC).to(device)  # Move model to GPU
    model_test.load_state_dict(torch.load(model_file_name_UPGA_J10_PC, map_location=device))  # Load to correct device
    
    Rtest, at, theta, ideal_beam = get_radar_data(snr_dB, H_test)
    # Move radar data to GPU
    Rtest = Rtest.to(device)
    at = at.to(device)
    theta = torch.as_tensor(theta, device=device)
    ideal_beam = torch.as_tensor(ideal_beam, device=device)
    
    rate_iter_UPGA_J10_PC, beam_error_iter_UPGA_J10_PC, F_prop_UPGA_J10_PC, W_prop_UPGA_J10_PC = model_test.execute_PGA(H_test, Rtest,
                                                                                                            snr,
                                                                                                            n_iter_outer,
                                                                                                            n_iter_inner_J10)
    rate_UPGA_J10_PC = torch.mean(rate_iter_UPGA_J10_PC, dim=0)
    beam_error_UPGA_J10_PC = torch.mean(beam_error_iter_UPGA_J10_PC, dim=0)
    iter_number_UPGA_J10_PC = torch.arange(n_iter_outer + 1, device=rate_UPGA_J10_PC.device, dtype=rate_UPGA_J10_PC.dtype)

    ## Plot results
    plt.figure()
    plt.semilogy(iter_number_UPGA_J10_PC.detach().cpu().tolist(),
                 beam_error_UPGA_J10_PC.detach().cpu().tolist(), '-o')
    plt.xlabel('Number of iterations')
    plt.ylabel('Beampattern MSE')
    plt.title('Beampattern MSE vs Number of iterations for UPGA J=10 PC')
    plt.grid()

    plt.figure()
    plt.plot(iter_number_UPGA_J10_PC.detach().cpu().tolist(),
             rate_UPGA_J10_PC.detach().cpu().tolist(), '-o')
    plt.xlabel('Number of iterations')
    plt.ylabel('Average achievable rate (bps/Hz)')
    plt.title('Average achievable rate vs Number of iterations for UPGA J=10 PC')
    plt.grid()