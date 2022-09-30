import matplotlib.pyplot as plt
import numpy as np
import torch

# def get_prediction(nsc, batch_ind_full, y_control, itr=500):
#     y_hat_list = list()
#     for i in range(itr):
#         with torch.no_grad():
#             B_reduced = nsc.get_B_reduced(batch_ind_full)
#             y_hat = torch.matmul(B_reduced, y_control)
#             if torch.sum(torch.isinf(y_hat)).item() == 0:
#                 y_hat_list.append(y_hat)

#     y_hat_mat = torch.stack(y_hat_list, dim=-1)
#     y_hat_mat[torch.isinf(y_hat_mat)] = 0.0
#     y_hat2 = torch.mean(y_hat_mat, dim=-1)
#     return y_hat2


def get_prediction(nsc, batch_ind_full, y_control, itr=500):
    batch_ind_full = batch_ind_full.to(nsc.B.device)
    y_control = y_control.to(nsc.B.device)
    y_hat_list = list()
    for i in range(itr):
        with torch.no_grad():
            B_reduced = nsc.get_B_reduced(batch_ind_full)
            y_hat = torch.matmul(B_reduced, y_control)
            if torch.sum(torch.isinf(y_hat)).item() == 0:
                y_hat_list.append(y_hat)

    y_hat_mat = torch.stack(y_hat_list, dim=-1)
    y_hat_mat[torch.isinf(y_hat_mat)] = 0.0
    y_hat2 = torch.mean(y_hat_mat, dim=-1)
    return y_hat2


def get_treatment_effect(nsc, batch_ind_full, y_full, y_control, itr=500):
    batch_ind_full = batch_ind_full.to(nsc.B.device)
    y_control = y_control.to(nsc.B.device)
    y_full = y_full.to(nsc.B.device)
    y_hat_list = list()
    for i in range(itr):
        with torch.no_grad():
            B_reduced = nsc.get_B_reduced(batch_ind_full)
            y_hat = torch.matmul(B_reduced, y_control)
            if torch.sum(torch.isinf(y_hat)).item() == 0:
                y_hat_list.append(y_hat)

    y_hat_mat = torch.stack(y_hat_list, dim=-1)
    y_hat_mat[torch.isinf(y_hat_mat)] = 0.0
    y_hat2 = torch.mean(y_hat_mat, dim=-1)
    return (y_full - y_hat2)[:, nsc.n_unit :, :], y_hat2


def summarize_B(mat_b, show_plot=True):
    if show_plot:
        plt.imshow(mat_b, cmap="hot", interpolation="nearest")
    ind = np.sum(mat_b, axis=1) != 0
    mat_b = mat_b[ind]
    gini = np.mean(np.sum(mat_b * (1 - mat_b), axis=1))
    gini_sd = np.std(np.sum(mat_b * (1 - mat_b), axis=1)) / np.sqrt(mat_b.shape[0])

    wrong_proba = np.mean(np.sum(mat_b[:, (mat_b.shape[1] // 2) :], axis=1))
    wrong_proba_sd = np.std(np.sum(mat_b[:, (mat_b.shape[1] // 2) :], axis=1)) / np.sqrt(mat_b.shape[0])

    entropy = np.mean(np.sum(mat_b * np.log(mat_b + 1e-9), axis=1)) * -1
    entropy_sd = np.std(np.sum(mat_b * np.log(mat_b + 1e-9), axis=1)) / np.sqrt(mat_b.shape[0])
    mis_class = np.sum(np.argmax(mat_b, axis=1) > mat_b.shape[1] / 2) / mat_b.shape[0]
    mis_class_sd = np.sqrt(mis_class * (1 - mis_class) / mat_b.shape[0])
    non_zeros = np.mean(np.sum(mat_b > 0.05, axis=1))
    non_zeros_sd = np.std(np.sum(mat_b > 0.05, axis=1)) / np.sqrt(mat_b.shape[0])
    print("Gini {:.3f} ({:.3f})".format(gini, gini_sd))
    print("Ent {:.3f} ({:.3f})".format(entropy, entropy_sd))
    print("N Matched {:.3f} ({:.3f})".format(non_zeros, non_zeros_sd))
    print("MIS {:.3f} ({:.3f})".format(mis_class, mis_class_sd))
    print("Wrong proba {:.3f} ({:.3f})".format(wrong_proba, wrong_proba_sd))

    return gini, gini_sd, mis_class, mis_class_sd, non_zeros, non_zeros_sd


def effect_mae_from_w(mat_w, n_units, y_control, y_full, treatment_effect):
    sc_w = torch.tensor(mat_w).to(y_control)
    sc_y_hat = torch.matmul(y_control.squeeze(), sc_w.T).unsqueeze(-1)
    sc_effect_est = y_full[:, n_units:, :] - sc_y_hat
    sc_mae_effect = torch.mean(torch.abs(treatment_effect - sc_effect_est)).item()
    sc_mae_effect_sd = torch.std(torch.abs(treatment_effect - sc_effect_est)).item() / np.sqrt(mat_w.shape[0])
    return sc_mae_effect, sc_mae_effect_sd


def summary_simulation(
    nsc,
    B_best,
    x_full,
    t_full,
    mask_full,
    batch_ind_full,
    y_full,
    y_control,
    plot_path="plots/sync/unit-{}-dim-{}-{}.png",
):

    with torch.no_grad():
        C = nsc.get_representation(x_full, t_full, mask_full)
        x_hat = nsc.get_reconstruction(C, t_full, mask_full)
        B_reduced = nsc.get_B_reduced(batch_ind_full)
        y_hat = torch.matmul(B_reduced, y_control)

        C_hat = torch.matmul(B_reduced, nsc.C0)

        self_expressive_loss = nsc.self_expressive_loss(C, B_reduced).item()
        reconstruction_loss = nsc.reconstruction_loss(x_full, x_hat, mask_full).item()
        if B_best is not None:
            y_best = torch.matmul(B_best, y_control)

            C_best = torch.matmul(B_best, nsc.C0)
            self_expressive_best = nsc.self_expressive_loss(C, B_best).item()

    print("self_expressive_loss: ", self_expressive_loss)
    print("reconstruction_loss: ", reconstruction_loss)
    x_np = x_full.cpu().numpy()
    y_np = y_full.cpu().numpy()
    y_hat = y_hat.cpu().numpy()
    x_hat = x_hat.cpu().numpy()
    C_np = C.cpu().numpy()
    C_hat = C_hat.cpu().numpy()
    if B_best is not None:
        y_best = y_best.cpu().numpy()
        C_best = C_best.cpu().numpy()

    # heatmap for matrix B
    plt.clf()
    plt.imshow(B_reduced.detach().cpu().numpy(), cmap="hot", interpolation="nearest")
    plt.savefig(plot_path.format(0, 0, "B-heatmap"))

    dim = 1
    for unit in range(12):

        # reconstruction
        plt.clf()
        plt.plot(x_np[:, unit, dim], label="true")
        plt.plot(x_hat[:, unit, dim], label="estimate")
        plt.legend()
        plt.savefig(plot_path.format(unit, dim, "reconstruction"))

        # treatment effect y
        plt.clf()
        plt.plot(y_np[:, unit, 0], label="true")
        plt.plot(y_hat[:, unit, 0], label="estimate")
        if B_best is not None:
            plt.plot(y_best[:, unit, 0], label="best")
        plt.legend()
        plt.savefig(plot_path.format(unit, dim, "treatment"))

        # B matrix
        plt.clf()
        with torch.no_grad():
            plt.plot(B_reduced.cpu().numpy()[unit], label="estimate")
            if B_best is not None:
                plt.plot(B_best.cpu().numpy()[unit], label="best")
            plt.legend()
            plt.savefig(plot_path.format(unit, dim, "B"))

        # representation
        plt.clf()
        plt.plot(C_np[unit], label="true")
        plt.plot(C_hat[unit], label="estimate")
        if B_best is not None:
            plt.plot(C_best[unit], label="best")
        plt.legend()
        plt.savefig(plot_path.format(unit, dim, "C"))
