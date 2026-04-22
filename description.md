I have a DNN to train the step sizes which is requred for a machine optimization.

if run_UPGA_J10 == 1:

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
            rate, __, F, W = model_UPGA_J10.execute_PGA(H, xi_0, A_dot, R_N_inv, snr_train, n_iter_outer,
                                                        n_iter_inner_J10)
            loss = get_sum_loss(F, W, H, xi_0, A_dot, R_N_inv, snr_train)
            # loss = -sum(sum(rate[1:]) / (K * batch_size))

            optimizer.zero_grad()  # zero the gradient buffers
            loss.backward()
            optimizer.step()  # Does the update

    # Save trained model
    torch.save(model_UPGA_J10.state_dict(), model_file_name_UPGA_J10)


class PGA_Unfold_J10(nn.Module):

    def __init__(self, step_size):
        super().__init__()
        self.step_size = nn.Parameter(step_size)  # parameters = (mu, lambda)

    # =========== Projection Gradient Ascent execution ===================
    def execute_PGA(self, H, xi_0, A_dot, R_N_inv, Pt, n_iter_outer, n_iter_inner):
        rate_init, F, W = initialize(H, Pt, initial_normalization)
        rate_over_iters = torch.zeros(n_iter_outer, len(H[0]))# save rates over iterations
        crb_over_iters = torch.zeros(n_iter_outer, len(H[0]))# save CRB over iterations

        for ii in range(n_iter_outer):
            # update F over
            for jj in range(n_iter_inner):
                grad_F_com = get_grad_F_com(H, F, W)
                grad_F_crb = get_grad_F_crb(F, W, xi_0, A_dot, R_N_inv)
                if grad_F_com.isnan().any() or grad_F_crb.isnan().any(): # check gradient
                    print('Error NaN gradients!!!!!!!!!!!!!!!')
                delta_F_com = self.step_size[jj][ii][0] * grad_F_com
                delta_F_crb = self.step_size[jj][ii][0] * grad_F_crb
                F = F + delta_F_com * WEIGHT_F_COM + delta_F_crb * WEIGHT_F_CRB
                # normalize by power to ensure non-NaN gradients if F becomes too large
                #if sum(torch.abs(F[0, :, 0, 0])) > 1e3:
                F = normalize_power(F, W, H, Pt)
            # Projection
            F = project_unit_modulus(F)

            # update W
            W_new = W.clone().detach()
            # compute gradients
            grad_W_k_com = get_grad_W_com(H, F, W)
            grad_W_k_crb = get_grad_W_crb(F, W, xi_0, A_dot, R_N_inv)
            for k in range(K):
                delta_W_com = self.step_size[0][ii][k + 1] * grad_W_k_com[k]
                delta_W_crb = self.step_size[0][ii][k + 1] * grad_W_k_crb[k]
                W_new[k] = W[k].clone().detach() + delta_W_com * WEIGHT_W_COM + delta_W_crb * WEIGHT_W_CRB
            # Projection
            F, W = normalize(F, W_new, H, Pt)

            # get the rate in this iteration
            rate_over_iters[ii] = get_sum_rate(H, F, W, Pt)
            # print(rate_over_iters[ii])
            rates = torch.cat([rate_over_iters], dim=0)
            crb_over_iters[ii] = get_crb_fe(H, F, W, xi_0, A_dot, R_N_inv, Pt)
            crb_fes = torch.cat([crb_over_iters], dim=0)

        return torch.transpose(rates, 0, 1), torch.transpose(crb_fes, 0, 1), F, W

But still I have used scalar step sizes for step size of the grad_F (mu) and step size of the grad_W (lambda). 

# ========================== initiate step sizes as tensor for training ================
    step_size_fixed = 1e-2  # step size of conventional PGA
    step_size_conv_PGA = torch.full([n_iter_outer, K + 1], step_size_fixed, requires_grad=True)
    step_size_UPGA_J1 = torch.full([n_iter_outer, K + 1], step_size_fixed, requires_grad=True)
    step_size_UPGA_J10 = torch.full([n_iter_inner_J10, n_iter_outer, K + 1], step_size_fixed, requires_grad=True)
    step_size_UPGA_J20 = torch.full([n_iter_inner_J20, n_iter_outer, K + 1], step_size_fixed, requires_grad=True)
    step_size_UPGA_J10_PC = torch.full([n_iter_inner_J10, n_iter_outer, K + 1], step_size_fixed, requires_grad=True)


Now I want to use preconditioning ( a matrix) as mu and lambda instead of scalar values. Refer the attached pdf for preconditioning. Initialy need to implement the basic version without 2nd-order information (AdaGrad, RMSProp, AdaDelta, Adam). 

I will select column wise preconditioning. Hence I hope mu is 64x64 matrix and lambda is 4x4 matrix.

1. As in the current setup I want to train the step size matrixes from the training loop.
2. The trained step sizes should be stored to use in the inference mode.
3. At the inference mode use the already optimized matrixes.

Give me the suggestions to modify the code.
