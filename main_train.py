import numpy as np
import matplotlib.pyplot as plt
import inspect
from utility import *
from PGA_models import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

TRAIN_LR = 5e-5
TRAIN_SCHEDULER_FACTOR = 0.5
TRAIN_SCHEDULER_PATIENCE = 3
TRAIN_MIN_LR = 1e-7
TRAIN_GRAD_CLIP_MAX_NORM = 1.0


def build_optimizer_and_scheduler(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
    scheduler_kwargs = {
        'mode': 'min',
        'factor': TRAIN_SCHEDULER_FACTOR,
        'patience': TRAIN_SCHEDULER_PATIENCE,
        'min_lr': TRAIN_MIN_LR,
    }
    if 'verbose' in inspect.signature(torch.optim.lr_scheduler.ReduceLROnPlateau.__init__).parameters:
        scheduler_kwargs['verbose'] = True

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        **scheduler_kwargs
    )
    return optimizer, scheduler


def clip_gradients(model):
    torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_GRAD_CLIP_MAX_NORM)

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
    optimizer, scheduler = build_optimizer_and_scheduler(model_UPGA_J1)
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
            clip_gradients(model_UPGA_J1)
            optimizer.step()  # Does the update

        scheduler.step(loss.item())

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
    model_UPGA_J20 = PGA_Unfold_JX(step_size_UPGA_J20)
    optimizer, scheduler = build_optimizer_and_scheduler(model_UPGA_J20)

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
            clip_gradients(model_UPGA_J20)
            optimizer.step()
            
            # .item() is critical to keep memory usage low!
            batch_losses.append(loss.item())

        avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg_loss)
        scheduler.step(avg_loss)
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
    model_UPGA_J10 = PGA_Unfold_JX(step_size_UPGA_J10)
    optimizer, scheduler = build_optimizer_and_scheduler(model_UPGA_J10)

    epoch_losses = [] # To store average loss per epoch

    for i_epoch in range(n_epoch):
        batch_losses = [] # To store loss of each batch in current epoch
        
        H_shuffled = torch.transpose(H_train, 0, 1)[np.random.permutation(len(H_train[0]))]
        print(f'size of the H_train: {H_train.shape}, size of the shuffled H: {H_shuffled.shape}')
        
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
            clip_gradients(model_UPGA_J10)
            optimizer.step()
            
            # .item() is critical to keep memory usage low!
            batch_losses.append(loss.item())

        avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg_loss)
        scheduler.step(avg_loss)
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

# ============================================================= proposed unfolding PGA =================================

if run_UPGA_J5 == 1:
    model_UPGA_J5 = PGA_Unfold_JX(step_size_UPGA_J5)
    optimizer, scheduler = build_optimizer_and_scheduler(model_UPGA_J5)

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
            
            rate, __, __, F, W = model_UPGA_J5.execute_PGA(H, xi_0, A_dot, R_N_inv, snr_train, n_iter_outer, n_iter_inner_J5, track_metrics=False)
            
            loss = get_sum_loss(F, W, H, xi_0, A_dot, R_N_inv, snr_train)
            print(f"Batch [{i_batch//batch_size+1}/{len(H_train[0])//batch_size}], Loss: {loss.item():.4f}")
            
            optimizer.zero_grad()
            loss.backward()
            clip_gradients(model_UPGA_J5)
            optimizer.step()
            
            # .item() is critical to keep memory usage low!
            batch_losses.append(loss.item())

        avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg_loss)
        scheduler.step(avg_loss)
        print(f"Epoch [{i_epoch+1}/{n_epoch}], Average Loss: {avg_loss:.4f}")

    torch.save(model_UPGA_J5.state_dict(), model_file_name_UPGA_J5)
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_epoch + 1), epoch_losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.savefig(directory_data + "training_loss_UPGA_J5.png")

if run_UPGA_J10_PRCDN == 1:
    model_UPGA_J10_PRCDN = PGA_Unfold_J10_PRCDN(n_iter_inner_J10, n_iter_outer, dim_F=64, dim_W=4).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model_UPGA_J10_PRCDN)

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
            clip_gradients(model_UPGA_J10_PRCDN)
            optimizer.step()
            
            # .item() is critical to keep memory usage low!
            batch_losses.append(loss.item())

        avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg_loss)
        scheduler.step(avg_loss)
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
    plt.savefig(directory_data + "training_loss_UPGA_J10.png")

# ============================================================= proposed unfolding PGA with decaying inner iterations ====
if run_UPGA_J5_decay == 1:
    model_UPGA_J5_decay = PGA_Unfold_JX_decay(step_size_UPGA_J5_decay)
    optimizer, scheduler = build_optimizer_and_scheduler(model_UPGA_J5_decay)

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

            __, __, __, F, W = model_UPGA_J5_decay.execute_PGA(
                H, xi_0, A_dot, R_N_inv, snr_train, n_iter_outer, n_iter_inner_J5, track_metrics=False)

            loss = get_sum_loss(F, W, H, xi_0, A_dot, R_N_inv, snr_train)
            print(f"Batch [{i_batch//batch_size+1}/{len(H_train[0])//batch_size}], Loss: {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            clip_gradients(model_UPGA_J5_decay)
            optimizer.step()

            batch_losses.append(loss.item())

        avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg_loss)
        scheduler.step(avg_loss)
        print(f"Epoch [{i_epoch+1}/{n_epoch}], Average Loss: {avg_loss:.4f}")

    torch.save(model_UPGA_J5_decay.state_dict(), model_file_name_UPGA_J5_decay)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_epoch + 1), epoch_losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss per Epoch (J decay)')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.savefig(directory_data + "training_loss_UPGA_J5_decay.png")

if run_UPGA_J10_decay == 1:
    model_UPGA_J10_decay = PGA_Unfold_JX_decay(step_size_UPGA_J10_decay)
    optimizer, scheduler = build_optimizer_and_scheduler(model_UPGA_J10_decay)

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

            __, __, __, F, W = model_UPGA_J10_decay.execute_PGA(
                H, xi_0, A_dot, R_N_inv, snr_train, n_iter_outer, n_iter_inner_J10, track_metrics=False)

            loss = get_sum_loss(F, W, H, xi_0, A_dot, R_N_inv, snr_train)
            print(f"Batch [{i_batch//batch_size+1}/{len(H_train[0])//batch_size}], Loss: {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            clip_gradients(model_UPGA_J10_decay)
            optimizer.step()

            batch_losses.append(loss.item())

        avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg_loss)
        scheduler.step(avg_loss)
        print(f"Epoch [{i_epoch+1}/{n_epoch}], Average Loss: {avg_loss:.4f}")

    torch.save(model_UPGA_J10_decay.state_dict(), model_file_name_UPGA_J10_decay)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_epoch + 1), epoch_losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss per Epoch (J decay)')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.savefig(directory_data + "training_loss_UPGA_J_decay.png")

# ============================================================= proposed unfolding PGA with decaying inner iterations (J_max=20) ====
if run_UPGA_J20_decay == 1:
    model_UPGA_J20_decay = PGA_Unfold_JX_decay(step_size_UPGA_J20_decay)
    optimizer, scheduler = build_optimizer_and_scheduler(model_UPGA_J20_decay)

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

            __, __, __, F, W = model_UPGA_J20_decay.execute_PGA(
                H, xi_0, A_dot, R_N_inv, snr_train, n_iter_outer, n_iter_inner_J20, track_metrics=False)

            loss = get_sum_loss(F, W, H, xi_0, A_dot, R_N_inv, snr_train)
            print(f"Batch [{i_batch//batch_size+1}/{len(H_train[0])//batch_size}], Loss: {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            clip_gradients(model_UPGA_J20_decay)
            optimizer.step()

            batch_losses.append(loss.item())

        avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg_loss)
        scheduler.step(avg_loss)
        print(f"Epoch [{i_epoch+1}/{n_epoch}], Average Loss: {avg_loss:.4f}")

    torch.save(model_UPGA_J20_decay.state_dict(), model_file_name_UPGA_J20_decay)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_epoch + 1), epoch_losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss per Epoch (J20 decay)')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.savefig(directory_result + "training_loss_UPGA_J20_decay.png")

# =========================================================== Unfolded PGA with gradient reuse ====
if run_UPGA_J_GradReuse == 1:
    model_UPGA_J_GradReuse = PGA_Unfold_J_GradReuse(step_size_UPGA_J_GradReuse)
    optimizer, scheduler = build_optimizer_and_scheduler(model_UPGA_J_GradReuse)

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
            clip_gradients(model_UPGA_J_GradReuse)
            optimizer.step()

            batch_losses.append(loss.item())

        avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg_loss)
        scheduler.step(avg_loss)
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
