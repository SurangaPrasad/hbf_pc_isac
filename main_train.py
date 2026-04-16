import numpy as np
import matplotlib.pyplot as plt
from utility import *
from PGA_models import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---- training and test the models ----

# Load training data
H_train, H_test0 = get_data_tensor(data_source)
print(H_train.shape)
H_test = H_test0[:, :test_size, :, :]
torch.manual_seed(3407)

# ====================================================== Conventional PGA ====================================
if run_conv_PGA == 1:
    # Object defining
    model_conv_PGA = PGA_Conv(step_size_conv_PGA)

    # executing classical PGA on the test set
    R, at, theta, ideal_beam = get_radar_data(snr_dB, H_test)

    rate_iter_conv, beam_iter_conv, F_conv, W_conv = model_conv_PGA.execute_PGA(H_test, R, snr, n_iter_outer)
    rate_conv = [r.detach().cpu().numpy() for r in (sum(rate_iter_conv) / len(H_test[0]))]
    beam_error_conv = [e.detach().cpu().numpy() for e in (sum(beam_iter_conv) / (len(H_test[0])))]
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
            cur_bs = H.shape[1]  # actual batch size (last mini-batch may be smaller)
            snr_dB_train = np.random.permutation(np.tile(snr_dB_list, batch_size // len(snr_dB_list)))[:cur_bs]  # balanced per-SNR
            snr_train = torch.tensor(10 ** (snr_dB_train / 10),
                                     dtype=torch.float32, device=device)        # (B,) tensor
            Rtrain, _, _, _ = get_radar_data(snr_dB_train, H)
            __, __, F, W = model_UPGA_J1.execute_PGA(H, Rtrain, snr_train, n_iter_outer, track_metrics=False)
            loss = get_sum_loss(F, W, H, Rtrain, snr_train, batch_size)

            optimizer.zero_grad()  # zero the gradient buffers
            loss.backward()
            optimizer.step()  # Does the update

    # Save trained model
    torch.save(model_UPGA_J1.state_dict(), model_file_name_UPGA_J1)

    # Create new model and load states
    model_test = PGA_Conv(step_size_UPGA_J1)
    model_test.load_state_dict(torch.load(model_file_name_UPGA_J1, map_location=device))

    # executing unfolded PGA on the test set
    Rtest, _, _, _ = get_radar_data(snr_dB, H_test)
    rate_iter_UPGA_J1, beam_error_iter_UPGA_J1, F_UPGA_J1, W_UPGA_J1 = model_test.execute_PGA(H_test, Rtest, snr, n_iter_outer)
    rate_UPGA_J1 = [r.detach().cpu().numpy() for r in (sum(rate_iter_UPGA_J1) / len(H_test[0]))]
    beam_error_UPGA_J1 = [r.detach().cpu().numpy() for r in (sum(beam_error_iter_UPGA_J1) / len(H_test[0]))]
    iter_number_UPGA_J1 = np.array(list(range(n_iter_outer + 1)))

# ============================================================= proposed unfolding PGA =================================
if run_UPGA_J20 == 1:
    model_UPGA_J20 = PGA_Unfold_J20(step_size_UPGA_J20)
    optimizer = torch.optim.Adam(model_UPGA_J20.parameters(), lr=learning_rate)

    epoch_losses = [] # To store average loss per epoch

    for i_epoch in range(n_epoch):
        batch_losses = [] # To store loss of each batch in current epoch
        
        H_shuffled = torch.transpose(H_train, 0, 1)[np.random.permutation(len(H_train[0]))]
        
        for i_batch in range(0, len(H_train[0]), batch_size):
            H = torch.transpose(H_shuffled[i_batch:i_batch + batch_size], 0, 1)
            cur_bs = H.shape[1]
            snr_dB_train = np.random.permutation(np.tile(snr_dB_list, batch_size // len(snr_dB_list)))[:cur_bs]  # balanced per-SNR
            snr_train = torch.tensor(10 ** (snr_dB_train / 10),
                                     dtype=torch.float32, device=device)
            
            rate, __, __, F, W = model_UPGA_J20.execute_PGA(H, xi_0, A_dot, R_N_inv, snr_train, n_iter_outer, n_iter_inner_J20, track_metrics=False)
            
            loss = get_sum_loss(F, W, H, xi_0, A_dot, R_N_inv, snr_train)
            print(f"Batch [{i_batch//batch_size+1}/{len(H_train[0])//batch_size}], Loss: {loss.item():.4f}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # .item() is critical to keep memory usage low!
            batch_losses.append(loss.item())

        avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{i_epoch+1}/{n_epoch}], Average Loss: {avg_loss:.4f}")

    torch.save(model_UPGA_J20.state_dict(), model_file_name_UPGA_J20)
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_epoch + 1), epoch_losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)


    # Save the plot of training loss
    plt.savefig(directory_data + "training_loss_UPGA_J20.png")

# ============================================================= proposed unfolding PGA =================================
if run_UPGA_J10 == 1:
    model_UPGA_J10 = PGA_Unfold_J10(step_size_UPGA_J10)
    optimizer = torch.optim.Adam(model_UPGA_J10.parameters(), lr=learning_rate)

    epoch_losses = [] # To store average loss per epoch

    for i_epoch in range(n_epoch):
        batch_losses = [] # To store loss of each batch in current epoch
        
        H_shuffled = torch.transpose(H_train, 0, 1)[np.random.permutation(len(H_train[0]))]
        
        for i_batch in range(0, len(H_train[0]), batch_size):
            H = torch.transpose(H_shuffled[i_batch:i_batch + batch_size], 0, 1)
            cur_bs = H.shape[1]
            snr_dB_train = np.random.permutation(np.tile(snr_dB_list, batch_size // len(snr_dB_list)))[:cur_bs]  # balanced per-SNR
            snr_train = torch.tensor(10 ** (snr_dB_train / 10),
                                     dtype=torch.float32, device=device)
            
            rate, __, __, F, W = model_UPGA_J10.execute_PGA(H, xi_0, A_dot, R_N_inv, snr_train, n_iter_outer, n_iter_inner_J10, track_metrics=False)
            
            loss = get_sum_loss(F, W, H, xi_0, A_dot, R_N_inv, snr_train)
            print(f"Batch [{i_batch//batch_size+1}/{len(H_train[0])//batch_size}], Loss: {loss.item():.4f}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # .item() is critical to keep memory usage low!
            batch_losses.append(loss.item())

        avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{i_epoch+1}/{n_epoch}], Average Loss: {avg_loss:.4f}")

    torch.save(model_UPGA_J10.state_dict(), model_file_name_UPGA_J10)
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_epoch + 1), epoch_losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)


    # Save the plot of training loss
    plt.savefig(directory_data + "training_loss_UPGA_J10.png")

    

    # # test proposed model
    # model_test = PGA_Unfold_J10(step_size_UPGA_J10)
    # model_test.load_state_dict(torch.load(model_file_name_UPGA_J10))
    # Rtest, at, theta, ideal_beam = get_radar_data(snr_dB, H_test)
    # rate_iter_UPGA_J10, beam_crb_iter_UPGA_J10, F_prop_UPGA_J10, W_prop_UPGA_J10 = model_test.execute_PGA(H_test, xi_0, A_dot, R_N_inv, snr,
    #                                                                                                         n_iter_outer,
    #                                                                                                         n_iter_inner_J10)
    # rate_UPGA_J10 = [r.detach().numpy() for r in (sum(rate_iter_UPGA_J10) / len(H_test[0]))]
    # beam_error_UPGA_J10 = [r.detach().numpy() for r in (sum(beam_crb_iter_UPGA_J10) / (len(H_test[0])))]
    # iter_number_UPGA_J10 = np.array(list(range(n_iter_outer + 1)))



    # Object defining
    # model_UPGA_J10_PC_AP = PGA_Unfold_J10_PC_AP(step_size_UPGA_J10_PC)

    # # training procedure
    # optimizer = torch.optim.Adam(model_UPGA_J10_PC_AP.parameters(), lr=learning_rate)

    # train_losses, valid_losses = [], []

    # for i_epoch in range(n_epoch):
    #     print(i_epoch)
    #     H_shuffeld = torch.transpose(H_train, 0, 1)[np.random.permutation(len(H_train[0]))]
    #     for i_batch in range(0, len(H_train), batch_size):
    #         H = torch.transpose(H_shuffeld[i_batch:i_batch + batch_size], 0, 1)
    #         cur_bs = H.shape[1]
    #         snr_dB_train = np.random.choice(snr_dB_list, size=cur_bs)
    #         snr_train = torch.tensor(10 ** (snr_dB_train / 10),
    #                                  dtype=torch.float32, device=device)
    #         Rtrain, _, _, _ = get_radar_data(snr_dB_train, H)
    #         __ , __, F, W = model_UPGA_J10_PC_AP.execute_PGA(H, Rtrain, snr_train, n_iter_outer, n_iter_inner_J10, track_metrics=False)
    #         loss = get_sum_loss(F, W, H, Rtrain, snr_train, batch_size)


    #         optimizer.zero_grad()  # zero the gradient buffers
    #         loss.backward()
    #         optimizer.step()  # Does the update

    # # Save trained model
    # torch.save(model_UPGA_J10_PC_AP.state_dict(), model_file_name_UPGA_J10_PC)


if run_UPGA_J10_PRCDN == 1:
    model_UPGA_J10_PRCDN = PGA_Unfold_J10_PRCDN(n_iter_inner_J10, n_iter_outer, dim_F=64, dim_W=4).to(device)
    optimizer = torch.optim.Adam(model_UPGA_J10_PRCDN.parameters(), lr=learning_rate)

    epoch_losses = [] # To store average loss per epoch

    for i_epoch in range(n_epoch):
        batch_losses = [] # To store loss of each batch in current epoch
        
        H_shuffled = torch.transpose(H_train, 0, 1)[np.random.permutation(len(H_train[0]))]
        
        for i_batch in range(0, len(H_train[0]), batch_size):
            H = torch.transpose(H_shuffled[i_batch:i_batch + batch_size], 0, 1)
            cur_bs = H.shape[1]
            snr_dB_train = np.random.permutation(np.tile(snr_dB_list, batch_size // len(snr_dB_list)))[:cur_bs]  # balanced per-SNR
            snr_train = torch.tensor(10 ** (snr_dB_train / 10),
                                     dtype=torch.float32, device=device)
            
            rate, __, F, W = model_UPGA_J10_PRCDN.execute_PGA(H, xi_0, A_dot, R_N_inv, snr_train, n_iter_outer, n_iter_inner_J10, track_metrics=False)
            
            loss = get_sum_loss(F, W, H, xi_0, A_dot, R_N_inv, snr_train)
            print(f"Batch [{i_batch//batch_size+1}/{len(H_train[0])//batch_size}], Loss: {loss.item():.4f}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # .item() is critical to keep memory usage low!
            batch_losses.append(loss.item())

        avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{i_epoch+1}/{n_epoch}], Average Loss: {avg_loss:.4f}")

    torch.save(model_UPGA_J10_PRCDN.state_dict(), model_file_name_UPGA_J10_PRCDN)
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_epoch + 1), epoch_losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)


    # Save the plot of training loss
    plt.savefig(directory_result + "training_loss_UPGA_J10.png")

# ============================================================= proposed unfolding PGA with learnable halting controller ====
if run_UPGA_J_decay == 1:
    model_UPGA_J_decay = PGA_Unfold_J_decay(step_size_UPGA_J_decay,
                                             hidden=STEP_CTRL_HIDDEN)
    optimizer = torch.optim.Adam(model_UPGA_J_decay.parameters(), lr=learning_rate)

    epoch_losses = []  # store average loss per epoch

    for i_epoch in range(n_epoch):
        batch_losses = []  # loss of each batch in current epoch

        H_shuffled = torch.transpose(H_train, 0, 1)[np.random.permutation(len(H_train[0]))]

        for i_batch in range(0, len(H_train[0]), batch_size):
            H = torch.transpose(H_shuffled[i_batch:i_batch + batch_size], 0, 1)
            cur_bs = H.shape[1]
            snr_dB_train = np.random.permutation(np.tile(snr_dB_list, batch_size // len(snr_dB_list)))[:cur_bs]
            snr_train = torch.tensor(10 ** (snr_dB_train / 10),
                                     dtype=torch.float32, device=device)

            # Soft halting (default): all max_inner steps run, weighted by halting probs
            __, __, __, F, W = model_UPGA_J_decay.execute_PGA(
                H, xi_0, A_dot, R_N_inv, snr_train, n_iter_outer,
                track_metrics=False, hard_halt=False)

            # Task loss + efficiency penalty (lambda * avg inner steps per outer iter).
            # Gradient flows: task_loss → F_final → gate_jj → j_soft → controller.
            # So the controller learns to halt when F quality is already good.
            task_loss = get_sum_loss(F, W, H, xi_0, A_dot, R_N_inv, snr_train)
            loss = task_loss + HALT_LAMBDA * model_UPGA_J_decay.total_j_soft

            # Per-iter stats: show step distribution across outer iterations
            j_arr = model_UPGA_J_decay.j_soft_per_iter.detach()
            print(f"Batch [{i_batch//batch_size+1}/{len(H_train[0])//batch_size}], "
                  f"Loss: {loss.item():.4f}, "
                  f"Steps min/mean/max: {j_arr.min().item():.1f}/{j_arr.mean().item():.1f}/{j_arr.max().item():.1f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{i_epoch+1}/{n_epoch}], Average Loss: {avg_loss:.4f}")

    torch.save(model_UPGA_J_decay.state_dict(), model_file_name_UPGA_J_decay)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_epoch + 1), epoch_losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss per Epoch (J decay - Learnable Halting)')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.savefig(directory_result + "training_loss_UPGA_J_decay.png")

# =========================================================== Unfolded PGA with gradient reuse ====
if run_UPGA_J_GradReuse == 1:
    model_UPGA_J_GradReuse = PGA_Unfold_J_GradReuse(step_size_UPGA_J_GradReuse)
    optimizer = torch.optim.Adam(model_UPGA_J_GradReuse.parameters(), lr=learning_rate)

    epoch_losses = []  # store average loss per epoch

    for i_epoch in range(n_epoch):
        batch_losses = []  # loss of each batch in current epoch

        H_shuffled = torch.transpose(H_train, 0, 1)[np.random.permutation(len(H_train[0]))]

        for i_batch in range(0, len(H_train[0]), batch_size):
            H = torch.transpose(H_shuffled[i_batch:i_batch + batch_size], 0, 1)
            cur_bs = H.shape[1]
            snr_dB_train = np.random.permutation(np.tile(snr_dB_list, batch_size // len(snr_dB_list)))[:cur_bs]
            snr_train = torch.tensor(10 ** (snr_dB_train / 10),
                                     dtype=torch.float32, device=device)

            __, __, __, F, W = model_UPGA_J_GradReuse.execute_PGA(
                H, xi_0, A_dot, R_N_inv, snr_train, n_iter_outer, n_iter_inner_J10, track_metrics=False)

            loss = get_sum_loss(F, W, H, xi_0, A_dot, R_N_inv, snr_train)
            print(f"Batch [{i_batch//batch_size+1}/{len(H_train[0])//batch_size}], Loss: {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{i_epoch+1}/{n_epoch}], Average Loss: {avg_loss:.4f}")
        print(f"  [GradReuse total fallback recomputations this epoch: "
              f"{model_UPGA_J_GradReuse.grad_recalc_count}]")

    torch.save(model_UPGA_J_GradReuse.state_dict(), model_file_name_UPGA_J_GradReuse)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_epoch + 1), epoch_losses, marker='o', linestyle='-', color='g')
    plt.title('Training Loss per Epoch (J GradReuse)')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.savefig(directory_result + "training_loss_UPGA_J_GradReuse.png")
