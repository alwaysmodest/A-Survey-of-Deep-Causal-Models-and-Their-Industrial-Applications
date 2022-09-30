import torch
import torch.nn as nn
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping
from ignite.metrics import Loss

from global_config import DEVICE


class DirectOutcomeRegression(nn.Module):
    def __init__(self, n_confounder, n_cause, n_outcome, n_hidden, linear=False, weighted=False, device=DEVICE):
        super().__init__()
        assert n_outcome == 1

        self.weighted = weighted

        if linear:
            self.mlp = nn.Sequential(nn.Linear(n_confounder + n_cause, n_outcome, bias=False),).to(device)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(n_confounder + n_cause, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_outcome)
            ).to(device)

        self.device = device

    def forward(self, input_mat):  # pylint: disable=arguments-differ
        # confounder = confounder.to(self.device)
        # cause = cause.to(self.device)
        # input_mat = torch.cat([confounder, cause], dim=-1)
        if not self.weighted:
            return self.mlp(input_mat)
        else:
            x = input_mat[:, :-1]
            w = input_mat[:, -1].unsqueeze(-1)
            return self.mlp(x), w

    def loss(self, y_pred, y):
        if not self.weighted:
            rmse = nn.MSELoss()
            return rmse(y_pred, y)
        else:
            y_pred, w = y_pred
            assert w.dim() == 2
            assert y_pred.dim() == 2
            assert y.dim() == 2

            return (w * (y_pred - y) ** 2).mean()


class NN_SCP(nn.Module):
    def __init__(
        self,
        single_cause_index,
        n_confounder,
        n_cause,
        n_outcome,
        n_confounder_rep,
        n_outcome_rep,
        mmd_sigma,
        lam_factual,
        lam_propensity,
        lam_mmd,
        linear=False,
        binary_outcome=False,
        device=DEVICE,
    ):
        super().__init__()
        self.single_cause_index = single_cause_index
        self.n_confounder = n_confounder
        self.n_cause = n_cause
        self.n_outcome = n_outcome
        self.mmd_sigma = mmd_sigma
        self.lam_factual = lam_factual
        self.lam_propensity = lam_propensity
        self.binary_outcome = binary_outcome
        self.lam_mmd = lam_mmd

        n_input = n_confounder + n_cause
        self.n_x = n_confounder + n_cause

        # outcome regression network

        if not self.binary_outcome:
            self.outcome_net0 = nn.Sequential(
                nn.Linear(n_input, n_confounder_rep + n_outcome_rep + 1),
                nn.ReLU(),
                nn.Linear(n_confounder_rep + n_outcome_rep + 1, n_outcome),
            ).to(device)

        else:
            # probability
            self.outcome_net0 = nn.Sequential(
                nn.Linear(n_input, n_confounder_rep + n_outcome_rep + 1),
                nn.ReLU(),
                nn.Linear(n_confounder_rep + n_outcome_rep + 1, n_outcome),
                nn.Sigmoid(),
            ).to(device)

    def forward(self, x):  # pylint: disable=arguments-differ
        outcome = self.outcome_net0(x)

        return outcome, 0, 0, 0

    def loss(self, y_pred, y):
        # y_pred is the output of forward
        # print('y_pred', len(y_pred))
        y_hat, _, _, _ = y_pred

        # factual loss
        if not self.binary_outcome:
            rmse = nn.MSELoss()
            # print('y_hat', y_hat.shape)
            # print('y', y.shape)
            error = torch.sqrt(rmse(y_hat, y))
        else:
            neg_y_hat = 1.0 - y_hat
            # N, 2, D_out
            y_hat_2d = torch.cat([y_hat[:, None, :], neg_y_hat[:, None, :]], dim=1)
            y_hat_2d = torch.log(y_hat_2d + 1e-9)
            outcome_nll_loss = nn.NLLLoss()
            # N, D_out
            y = y.to(torch.long)
            error = outcome_nll_loss(y_hat_2d, y)

        loss = error

        # print('error', error.item())
        # print('nll', nll.item())
        # print('mmd', mmd.item())

        return loss


class ModelTrainer:
    def __init__(self, batch_size, max_epoch, loss_fn, model_id, model_path="model/"):
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.model_id = model_id
        self.max_epoch = max_epoch
        self.model_path = model_path

    def train(self, model, optimizer, train_dataset, valid_dataset, print_every=1):

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)

        model.train()
        optimizer.zero_grad()
        trainer = create_supervised_trainer(model, optimizer, self.loss_fn)
        evaluator = create_supervised_evaluator(model, metrics={"loss": Loss(self.loss_fn)})

        # early stopping
        def score_function(engine):
            val_loss = engine.state.metrics["loss"]
            return -val_loss

        early_stopping_handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
        evaluator.add_event_handler(Events.COMPLETED, early_stopping_handler)

        # evaluation loss
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            if trainer.state.epoch % print_every == 0:
                print("Validation Results - Epoch[{}] Avg loss: {:.3f}".format(trainer.state.epoch, metrics["loss"]))

        # save best model
        save_best_model_by_val_score(
            self.model_path,
            evaluator,
            model,
            "loss",
            n_saved=1,
            score_fun=score_function,
            tag="val",
            model_id=self.model_id,
        )

        trainer.run(train_loader, max_epochs=self.max_epoch)

        return model


def gen_save_best_models_by_val_score(
    save_handler, evaluator, models, metric_name, n_saved=3, score_fun=None, tag="val", model_id="best", **kwargs
):
    """Method adds a handler to ``evaluator`` to save ``n_saved`` of best models based on the metric
    (named by ``metric_name``) provided by ``evaluator`` (i.e. ``evaluator.state.metrics[metric_name]``).
    Models with highest metric value will be retained. The logic of how to store objects is delegated to
    ``save_handler``.

    Args:
        save_handler (callable or :class:`~ignite.handlers.checkpoint.BaseSaveHandler`): Method or callable class to
            use to save engine and other provided objects. Function receives two objects: checkpoint as a dictionary
            and filename. If ``save_handler`` is callable class, it can
            inherit of :class:`~ignite.handlers.checkpoint.BaseSaveHandler` and optionally implement ``remove`` method
            to keep a fixed number of saved checkpoints. In case if user needs to save engine's checkpoint on a disk,
            ``save_handler`` can be defined with :class:`~ignite.handlers.DiskSaver`.
        evaluator (Engine): evaluation engine used to provide the score
        models (nn.Module or Mapping): model or dictionary with the object to save. Objects should have
            implemented ``state_dict`` and ``load_state_dict`` methods.
        metric_name (str): metric name to use for score evaluation. This metric should be present in
            `evaluator.state.metrics`.
        n_saved (int, optional): number of best models to store
        trainer (Engine, optional): trainer engine to fetch the epoch when saving the best model.
        tag (str, optional): score name prefix: `{tag}_{metric_name}`. By default, tag is "val".
        **kwargs: optional keyword args to be passed to construct :class:`~ignite.handlers.checkpoint.Checkpoint`.

    Returns:
        A :class:`~ignite.handlers.checkpoint.Checkpoint` handler.
    """
    global_step_transform = None

    to_save = models
    if isinstance(models, nn.Module):
        to_save = {"model": models}

    best_model_handler = Checkpoint(
        to_save,
        save_handler,
        filename_prefix=model_id,
        n_saved=n_saved,
        global_step_transform=global_step_transform,
        score_name="{}_{}".format(tag, metric_name.lower()),
        score_function=score_fun,
        **kwargs,
    )
    evaluator.add_event_handler(
        Events.COMPLETED, best_model_handler,
    )

    return best_model_handler


def save_best_model_by_val_score(
    output_path, evaluator, model, metric_name, n_saved=3, score_fun=None, tag="val", model_id="best", **kwargs
):
    """Method adds a handler to ``evaluator`` to save on a disk ``n_saved`` of best models based on the metric
    (named by ``metric_name``) provided by ``evaluator`` (i.e. ``evaluator.state.metrics[metric_name]``).
    Models with highest metric value will be retained.

    Args:
        output_path (str): output path to indicate where to save best models
        evaluator (Engine): evaluation engine used to provide the score
        model (nn.Module): model to store
        metric_name (str): metric name to use for score evaluation. This metric should be present in
            `evaluator.state.metrics`.
        n_saved (int, optional): number of best models to store
        trainer (Engine, optional): trainer engine to fetch the epoch when saving the best model.
        tag (str, optional): score name prefix: `{tag}_{metric_name}`. By default, tag is "val".
        **kwargs: optional keyword args to be passed to construct :class:`~ignite.handlers.checkpoint.Checkpoint`.

    Returns:
        A :class:`~ignite.handlers.checkpoint.Checkpoint` handler.
    """
    return gen_save_best_models_by_val_score(
        save_handler=DiskSaver(dirname=output_path, require_empty=False),
        evaluator=evaluator,
        models=model,
        metric_name=metric_name,
        n_saved=n_saved,
        score_fun=score_fun,
        tag=tag,
        model_id=model_id,
        **kwargs,
    )
