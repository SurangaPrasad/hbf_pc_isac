import numpy as np
import matplotlib.pyplot as plt
from utility import *
from PGA_models import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---- training and test the models ----

# Load training data
H_train = get_data_tensor(data_source)
# H_test = H_test0[:, :test_size, :, :]
torch.manual_seed(3407)

# ====================================================== Conventional PGA ====================================
if run_conv_PGA == 1:
    # Object defining
    model_conv_PGA = PGA_Conv(step_size_conv_PGA)

    # executing classical PGA on the test set
    R, at, theta, ideal_beam = get_radar_data(snr_dB, H_test)

    rate_iter_conv, beam_iter_conv, F_conv, W_conv = model_conv_PGA.execute_PGA(H_test, R, snr, n_iter_outer)
    rate_conv = [r.detach().numpy() for r in (sum(rate_iter_conv) / len(H_test[0]))]
    beam_error_conv = [e.detach().numpy() for e in (sum(beam_iter_conv) / (len(H_test[0])))]
    iter_number_conv = np.array(list(range(n_iter_outer + 1)))

# ====================================================== Unfolded PGA with J = 1 ====================================
if run_UPGA_J1 == 1:

    # Object defining
    model_UPGA_J1 = PGA_Conv(step_size_UPGA_J1)
    # training procedure
    optimizer = torch.optim.Adam(model_UPGA_J1.parameters(), lr=learning_rate)
    train_losses, valid_losses = [], []

    for i_epoch in range(n_epoch):
        print(i_epoch)
        H_shuffeld = torch.transpose(H_train, 0, 1)[np.random.permutation(len(H_train[0]))]
        for i_batch in range(0, len(H_train), batch_size):
            H = torch.transpose(H_shuffeld[i_batch:i_batch + batch_size], 0, 1)
            snr_dB_train = np.random.choice(snr_dB_list)
            snr_train = 10 ** (snr_dB_train / 10)
            Rtrain, _, _, _ = get_radar_data(snr_dB_train, H)
            __, __, F, W = model_UPGA_J1.execute_PGA(H, Rtrain, snr_train, n_iter_outer)
            loss = get_sum_loss(F, W, H, Rtrain, snr_train, batch_size)

            optimizer.zero_grad()  # zero the gradient buffers
            loss.backward()
            optimizer.step()  # Does the update

    # Save trained model
    torch.save(model_UPGA_J1.state_dict(), model_file_name_UPGA_J1)

    # Create new model and load states
    model_test = PGA_Conv(step_size_UPGA_J1)
    model_test.load_state_dict(torch.load(model_file_name_UPGA_J1))

    # executing unfolded PGA on the test set
    Rtest, _, _, _ = get_radar_data(snr_dB, H_test)
    rate_iter_UPGA_J1, beam_error_iter_UPGA_J1, F_UPGA_J1, W_UPGA_J1 = model_test.execute_PGA(H_test, Rtest, snr, n_iter_outer)
    rate_UPGA_J1 = [r.detach().numpy() for r in (sum(rate_iter_UPGA_J1) / len(H_test[0]))]
    beam_error_UPGA_J1 = [r.detach().numpy() for r in (sum(beam_error_iter_UPGA_J1) / len(H_test[0]))]
    iter_number_UPGA_J1 = np.array(list(range(n_iter_outer + 1)))

# ============================================================= proposed unfolding PGA =================================
if run_UPGA_J20 == 1:

    print(f'Running Unfolded PGA with J=20 ...')
    # Object defining
    model_UPGA_J20 = PGA_Unfold_J20(step_size_UPGA_J20)

    # training procedure
    optimizer = torch.optim.Adam(model_UPGA_J20.parameters(), lr=learning_rate)

    train_losses, valid_losses = [], []

    for i_epoch in range(n_epoch):
        print(i_epoch)
        H_shuffeld = torch.transpose(H_train, 0, 1)[np.random.permutation(len(H_train[0]))]
        for i_batch in range(0, len(H_train), batch_size):
            H = torch.transpose(H_shuffeld[i_batch:i_batch + batch_size], 0, 1)
            snr_dB_train = np.random.choice(snr_dB_list)
            snr_train = 10 ** (snr_dB_train / 10)
            Rtrain, _, _, _ = get_radar_data(snr_dB_train, H)
            print(f'Rtrain shape: {Rtrain.shape}')
            rate, __, F, W = model_UPGA_J20.execute_PGA(H, Rtrain, snr_train, n_iter_outer, n_iter_inner_J20)
            loss = get_sum_loss(F, W, H, Rtrain, snr_train, batch_size)
            # loss = -sum(sum(rate[1:]) / (K * batch_size))

            optimizer.zero_grad()  # zero the gradient buffers
            loss.backward()
            optimizer.step()  # Does the update

    # Save trained model
    torch.save(model_UPGA_J20.state_dict(), model_file_name_UPGA_J20)

    # test proposed model
    model_test = PGA_Unfold_J20(step_size_UPGA_J20)
    model_test.load_state_dict(torch.load(model_file_name_UPGA_J20))
    Rtest, at, theta, ideal_beam = get_radar_data(snr_dB, H_test)
    rate_iter_UPGA_J20, beam_error_iter_UPGA_J20, F_UPGA_J20, W_UPGA_J20 = model_test.execute_PGA(H_test, Rtest, snr, n_iter_outer,
                                                                                             n_iter_inner_J20)
    rate_UPGA_J20 = [r.detach().numpy() for r in (sum(rate_iter_UPGA_J20) / len(H_test[0]))]
    beam_error_UPGA_J20 = [r.detach().numpy() for r in (sum(beam_error_iter_UPGA_J20) / (len(H_test[0])))]
    iter_number_UPGA_J20 = np.array(list(range(n_iter_outer + 1)))

# ============================================================= proposed unfolding PGA =================================
if run_UPGA_J10 == 1:

    print(f'Running Unfolded PGA with J=10 ...')
    # Object defining
    model_UPGA_J10 = PGA_Unfold_J10(step_size_UPGA_J10)

    # training procedure
    optimizer = torch.optim.Adam(model_UPGA_J10.parameters(), lr=learning_rate)

    train_losses, valid_losses = [], []

    for i_epoch in range(n_epoch):
        print(i_epoch)
        H_shuffeld = torch.transpose(H_train, 0, 1)[np.random.permutation(len(H_train[0]))]
        for i_batch in range(0, len(H_train), batch_size):
            H = torch.transpose(H_shuffeld[i_batch:i_batch + batch_size], 0, 1)
            snr_dB_train = np.random.choice(snr_dB_list)
            snr_train = 10 ** (snr_dB_train / 10)
            Rtrain, _, _, _ = get_radar_data(snr_dB_train, H)
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
    model_test = PGA_Unfold_J10(step_size_UPGA_J10)
    model_test.load_state_dict(torch.load(model_file_name_UPGA_J10))
    Rtest, at, theta, ideal_beam = get_radar_data(snr_dB, H_test)
    rate_iter_UPGA_J10, beam_error_iter_UPGA_J10, F_prop_UPGA_J10, W_prop_UPGA_J10 = model_test.execute_PGA(H_test, Rtest,
                                                                                                            snr,
                                                                                                            n_iter_outer,
                                                                                                            n_iter_inner_J10)
    rate_UPGA_J10 = [r.detach().numpy() for r in (sum(rate_iter_UPGA_J10) / len(H_test[0]))]
    beam_error_UPGA_J10 = [r.detach().numpy() for r in (sum(beam_error_iter_UPGA_J10) / (len(H_test[0])))]
    iter_number_UPGA_J10 = np.array(list(range(n_iter_outer + 1)))


# ============================================================= RKD Distillation for UPGA J=10 =================================
if run_RKD_Distillation == 1:

    print(f'Running RKD Distillation for UPGA J=10 ...')
    # 1. Load Pre-trained Teacher (J=20)
    model_teacher = PGA_Unfold_J20(step_size_UPGA_J20)
    model_teacher.load_state_dict(torch.load(model_file_name_UPGA_J20))
    model_teacher.eval() # Teacher stays in evaluation mode

    # 2. Define Student (J=10)
    model_student = PGA_Unfold_J10(step_size_UPGA_J10)
    optimizer = torch.optim.Adam(model_student.parameters(), lr=learning_rate)
    
    # Weighting factors for RKD losses
    lambda_dist = 1.0
    lambda_angle = 2.0

    for i_epoch in range(n_epoch):
        print(i_epoch)
        H_shuffeld = torch.transpose(H_train, 0, 1)[np.random.permutation(len(H_train))]
        
        for i_batch in range(0, len(H_train), batch_size):
            H = torch.transpose(H_shuffeld[i_batch:i_batch + batch_size], 0, 1)
            snr_dB_train = np.random.choice(snr_dB_list)
            snr_train = 10 ** (snr_dB_train / 10)
            Rtrain, _, _, _ = get_radar_data(snr_dB_train, H)

            # --- Teacher Forward Pass (No Gradients) ---
            with torch.no_grad():
                _, _, F_t, W_t = model_teacher.execute_PGA(H, Rtrain, snr_train, n_iter_outer, n_iter_inner_J20)
                # Representation: Flatten F_t (Analog Precoder) to act as 'embedding'
                teacher_repr = F_t.view(batch_size, -1) 

            # --- Student Forward Pass ---
            rate, _, F_s, W_s = model_student.execute_PGA(H, Rtrain, snr_train, n_iter_outer, n_iter_inner_J10)
            student_repr = F_s.view(batch_size, -1)

            # --- Loss Calculation ---
            # 1. Original Task Loss (Source [9, 13])
            loss_task = get_sum_loss(F_s, W_s, H, Rtrain, snr_train, batch_size)
            
            # 2. RKD Losses (Source [6, 7, 9])
            loss_dist = rkd_distance_loss(teacher_repr, student_repr)
            loss_angle = rkd_angle_loss(teacher_repr, student_repr)
            
            # Total Loss = Task Loss + (lambda * RKD Loss) [9]
            total_loss = loss_task + (lambda_dist * loss_dist) + (lambda_angle * loss_angle)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    # Save trained student model
    torch.save(model_student.state_dict(), model_file_name_UPGA_J10 + '_RKD.pth')
