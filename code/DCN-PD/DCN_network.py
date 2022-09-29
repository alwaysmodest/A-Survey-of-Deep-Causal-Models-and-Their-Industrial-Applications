import torch
import torch.nn as nn
import torch.optim as optim

from DCN import DCN


class DCN_network:
    def train(self, train_parameters, device):
        epochs = train_parameters["epochs"]
        treated_batch_size = train_parameters["treated_batch_size"]
        control_batch_size = train_parameters["control_batch_size"]
        lr = train_parameters["lr"]
        shuffle = train_parameters["shuffle"]
        model_save_path = train_parameters["model_save_path"].format(epochs, lr)
        treated_set = train_parameters["treated_set"]
        control_set = train_parameters["control_set"]

        print("Saved model path: {0}".format(model_save_path))

        treated_data_loader = torch.utils.data.DataLoader(treated_set,
                                                          batch_size=treated_batch_size,
                                                          shuffle=shuffle,
                                                          num_workers=1)

        control_data_loader = torch.utils.data.DataLoader(control_set,
                                                          batch_size=control_batch_size,
                                                          shuffle=shuffle,
                                                          num_workers=1)
        network = DCN(training_flag=True).to(device)
        optimizer = optim.Adam(network.parameters(), lr=lr)
        lossF = nn.MSELoss()
        min_loss = 100000.0
        dataset_loss = 0.0
        print(".. Training started ..")
        print(device)
        for epoch in range(epochs):
            network.train()
            total_loss = 0
            train_set_size = 0

            if epoch % 2 == 0:
                dataset_loss = 0
                # train treated
                network.hidden1_Y1.weight.requires_grad = True
                network.hidden1_Y1.bias.requires_grad = True
                network.hidden2_Y1.weight.requires_grad = True
                network.hidden2_Y1.bias.requires_grad = True
                network.out_Y1.weight.requires_grad = True
                network.out_Y1.bias.requires_grad = True

                network.hidden1_Y0.weight.requires_grad = False
                network.hidden1_Y0.bias.requires_grad = False
                network.hidden2_Y0.weight.requires_grad = False
                network.hidden2_Y0.bias.requires_grad = False
                network.out_Y0.weight.requires_grad = False
                network.out_Y0.bias.requires_grad = False

                for batch in treated_data_loader:
                    covariates_X, ps_score, y_f, y_cf = batch
                    covariates_X = covariates_X.to(device)
                    ps_score = ps_score.squeeze().to(device)

                    train_set_size += covariates_X.size(0)
                    treatment_pred = network(covariates_X, ps_score)
                    # treatment_pred[0] -> y1
                    # treatment_pred[1] -> y0
                    predicted_ITE = treatment_pred[0] - treatment_pred[1]
                    true_ITE = y_f - y_cf
                    if torch.cuda.is_available():
                        loss = lossF(predicted_ITE.float().cuda(),
                                     true_ITE.float().cuda()).to(device)
                    else:
                        loss = lossF(predicted_ITE.float(),
                                     true_ITE.float()).to(device)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                dataset_loss = total_loss

            elif epoch % 2 == 1:
                # train controlled
                network.hidden1_Y1.weight.requires_grad = False
                network.hidden1_Y1.bias.requires_grad = False
                network.hidden2_Y1.weight.requires_grad = False
                network.hidden2_Y1.bias.requires_grad = False
                network.out_Y1.weight.requires_grad = False
                network.out_Y1.bias.requires_grad = False

                network.hidden1_Y0.weight.requires_grad = True
                network.hidden1_Y0.bias.requires_grad = True
                network.hidden2_Y0.weight.requires_grad = True
                network.hidden2_Y0.bias.requires_grad = True
                network.out_Y0.weight.requires_grad = True
                network.out_Y0.bias.requires_grad = True

                for batch in control_data_loader:
                    covariates_X, ps_score, y_f, y_cf = batch
                    covariates_X = covariates_X.to(device)
                    ps_score = ps_score.squeeze().to(device)

                    train_set_size += covariates_X.size(0)
                    treatment_pred = network(covariates_X, ps_score)
                    # treatment_pred[0] -> y1
                    # treatment_pred[1] -> y0
                    predicted_ITE = treatment_pred[0] - treatment_pred[1]
                    true_ITE = y_cf - y_f
                    if torch.cuda.is_available():
                        loss = lossF(predicted_ITE.float().cuda(),
                                     true_ITE.float().cuda()).to(device)
                    else:
                        loss = lossF(predicted_ITE.float(),
                                     true_ITE.float()).to(device)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                dataset_loss = dataset_loss + total_loss

            print("epoch: {0}, train_set_size: {1} loss: {2}".
                  format(epoch, train_set_size, total_loss))

            if epoch % 2 == 1:
                print("Treated + Control loss: {0}".format(dataset_loss))
                # if dataset_loss < min_loss:
                #     print("Current loss: {0}, over previous: {1}, Saving model".
                #           format(dataset_loss, min_loss))
                #     min_loss = dataset_loss
                #     torch.save(network.state_dict(), model_save_path)

        torch.save(network.state_dict(), model_save_path)

    @staticmethod
    def eval(eval_parameters, device):
        print(".. Evaluation started ..")
        treated_set = eval_parameters["treated_set"]
        control_set = eval_parameters["control_set"]
        model_path = eval_parameters["model_save_path"]
        network = DCN(training_flag=False).to(device)
        network.load_state_dict(torch.load(model_path, map_location=device))
        network.eval()
        treated_data_loader = torch.utils.data.DataLoader(treated_set,
                                                          shuffle=False, num_workers=1)
        control_data_loader = torch.utils.data.DataLoader(control_set,
                                                          shuffle=False, num_workers=1)

        err_treated_list = []
        err_control_list = []

        for batch in treated_data_loader:
            covariates_X, ps_score, y_f, y_cf = batch
            covariates_X = covariates_X.to(device)
            ps_score = ps_score.squeeze().to(device)
            treatment_pred = network(covariates_X, ps_score)

            predicted_ITE = treatment_pred[0] - treatment_pred[1]
            true_ITE = y_f - y_cf
            if torch.cuda.is_available():
                diff = true_ITE.float().cuda() - predicted_ITE.float().cuda()
            else:
                diff = true_ITE.float() - predicted_ITE.float()

            err_treated_list.append(diff.item())

        for batch in control_data_loader:
            covariates_X, ps_score, y_f, y_cf = batch
            covariates_X = covariates_X.to(device)
            ps_score = ps_score.squeeze().to(device)
            treatment_pred = network(covariates_X, ps_score)

            predicted_ITE = treatment_pred[0] - treatment_pred[1]
            true_ITE = y_cf - y_f
            if torch.cuda.is_available():
                diff = true_ITE.float().cuda() - predicted_ITE.float().cuda()
            else:
                diff = true_ITE.float() - predicted_ITE.float()
            err_control_list.append(diff.item())

        # print(err_treated_list)
        # print(err_control_list)
        return {
            "treated_err": err_treated_list,
            "control_err": err_control_list,
        }
