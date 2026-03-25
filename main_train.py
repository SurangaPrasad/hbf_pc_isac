import math
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from utility import *
from PGA_models import *
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
seed = 3407 
random.seed(seed) 
np.random.seed(seed) 
torch.manual_seed(seed) 
torch.cuda.manual_seed_all(seed)

# ---- GPU Setup: Detect and configure device ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
amp_enabled = torch.cuda.is_available()
enable_amp = torch.cuda.is_available()
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

amp_enabled = bool(enable_amp) and device.type == 'cuda'
scaler = GradScaler(enabled=amp_enabled)
train_micro_batch_size = max(1, batch_size)
if train_micro_batch_size != batch_size:
    print(f"Using micro-batches of {train_micro_batch_size} (logical batch {batch_size}).")
if amp_enabled:
    print("Mixed precision (torch.cuda.amp) is enabled.")

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
            print(f'Dimmensions of H: {H.shape}')
            _train_with_micro_batches(
                model_UPGA_J20,
                optimizer,
                H,
                Rtrain,
                snr_train,
                n_iter_outer,
                scaler,
                n_iter_inner=n_iter_inner_J20,
            )

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

# ============================================================= RKD Distillation for UPGA J=10 =================================
if run_RKD_Distillation == 1:
    
    print(f'Running RKD Distillation for UPGA J=10 ...')
    
    # Teacher
    model_teacher = PGA_Unfold_J20(step_size_UPGA_J20).to(device)
    model_teacher.load_state_dict(torch.load(directory_model+'UPGA_J20.pth', map_location=device))
    model_teacher.eval()

    # Student
    model_student = PGA_Unfold_J10(step_size_UPGA_J10).to(device)
    optimizer = torch.optim.Adam(model_student.parameters(), lr=learning_rate)

    lambda_dist = 25.0
    lambda_angle = 50.0

    for i_epoch in range(n_epoch):
        start_time = time.time()
        print(i_epoch)
        epoch_rkd = 0.0
        epoch_loss = 0.0
        epoch_task_student = 0.0
        epoch_task_teacher = 0.0
        num_batches = 0
        H_shuffeld = torch.transpose(H_train, 0, 1)[np.random.permutation(len(H_train[0]))]

        for i_batch in range(0, len(H_train[0]), batch_size):
            H = torch.transpose(H_shuffeld[i_batch:i_batch + batch_size], 0, 1).to(device)
            current_bs = H.shape[1]
            snr_dB_train = np.random.choice(snr_dB_list)
            snr_train = 10 ** (snr_dB_train / 10)

            Rtrain, _, _, _ = get_radar_data(snr_dB_train, H)
            Rtrain = Rtrain.to(device)

            # Teacher (no grad)
            with torch.no_grad():
                print(f'Calculating teacher outputs for batch {i_batch} ...')
                _, _, F_t, W_t = model_teacher.execute_PGA(H, Rtrain, snr_train, n_iter_outer, n_iter_inner_J20)
                #print(W_t.shape)
                T_rkd = torch.matmul(F_t, W_t).view(current_bs, -1)
                #print(T_rkd.shape)
                teacher_task_loss= get_sum_loss(F_t, W_t, H, Rtrain, snr_train, current_bs)
                #Normalize
                teacher_repr = torch.nn.functional.normalize(T_rkd, dim=1).detach()
                #print(teacher_repr.shape)
            # Student
            rate, _, F_s, W_s = model_student.execute_PGA(H, Rtrain, snr_train, n_iter_outer, n_iter_inner_J10)
            S_rkd = torch.matmul(F_s, W_s).view(current_bs, -1)
            #Normalize
            student_repr = torch.nn.functional.normalize(S_rkd, dim=1)
            #print(student_repr.shape)
            # Loss
            # print(f'Calculating losses for batch {i_batch} ...')
            loss_task = get_sum_loss(F_s, W_s, H, Rtrain, snr_train, current_bs)
            # print(f'Loss task: {loss_task.item():.4f}, Loss teacher: {teacher_task_loss.item():.4f}')
            loss_dist = rkd_distance_loss(teacher_repr, student_repr)
            # print(f'Loss dist: {loss_dist.item():.4f}')
            loss_angle = rkd_angle_loss(teacher_repr, student_repr)
            
            # print the loss for the batch nunber
            print(f'Batch {i_batch} | Loss task: {loss_task.item():.4f}, Loss dist: {loss_dist.item():.4f}, Loss angle: {loss_angle.item():.4f}')

            total_loss = loss_task + (lambda_dist * loss_dist + lambda_angle * loss_angle)
            #criterion_mse = torch.nn.MSELoss()
            #loss_mse = criterion_mse(teacher_task_loss, loss_task)
            #total_loss = loss_mse
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    
            epoch_loss += total_loss.item()
            epoch_rkd += (lambda_dist * loss_dist.item() + lambda_angle * loss_angle.item())
            epoch_task_student += loss_task.item()
            epoch_task_teacher += teacher_task_loss.item()
            num_batches += 1  
            end_time = time.time()     # ⏱️ end
            epoch_time = end_time - start_time
        
    
        print(
            f"Epoch {i_epoch} | "
            f"Time: {epoch_time:.2f}s | "
            f"RKD_dist: {epoch_rkd / num_batches:.4f} | "
            f"Avg Loss: {epoch_loss / num_batches:.4f} | "
            f"Student: {epoch_task_student / num_batches:.4f} | "
            f"Teacher: {epoch_task_teacher / num_batches:.4f}"
        )
    torch.save(model_student.state_dict(), model_file_name_UPGA_J10 + '_RKD_new_B16_all.pth')

    ## Plot results