import math

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from axolotl.utils.distributed import is_main_process
from axolotl.utils import is_comet_available
import logging

if is_comet_available():
    import comet_ml

LOG = logging.getLogger("axolotl.callbacks")


class AlphaSchedulerCallback(TrainerCallback):
    """
    Callback to adjust `alpha` in model.config dynamically during training
    """

    def __init__(self, cfg):
        print(">>>>>>>>>>>>>>>>>>>>>>>> Initializing AlphaSchedulerCallback")
        self.cfg = cfg
        initial_alpha = self.cfg.alpha_scheduler_initial_alpha
        final_alpha = self.cfg.alpha_scheduler_final_alpha
        max_steps = self.cfg.alpha_scheduler_max_steps
        strategy = self.cfg.alpha_scheduler_strategy
        power = self.cfg.alpha_scheduler_power  # For polynomial annealing
        step_size = self.cfg.alpha_scheduler_step_size  # For stepwise annealing
        increment = self.cfg.alpha_scheduler_increment  # For stepwise annealing
        k = self.cfg.alpha_scheduler_k  # For sigmoid annealing

        if initial_alpha is None:
            initial_alpha = self.cfg.adasplash_alpha
            LOG.warning("Initial alpha value not provided. Using: {}".format(initial_alpha))

        if final_alpha is None:
            final_alpha = self.cfg.adasplash_alpha
            LOG.warning("Final alpha value not provided. Using: {}".format(final_alpha))

        if max_steps is None:
            if self.cfg.max_steps is not None:
                max_steps = self.cfg.max_steps
            else:
                max_steps = 10000
            LOG.warning("Max steps not provided. Using: {}".format(max_steps))

        if strategy is None:
            strategy = "linear"
            LOG.warning("Annealing strategy not provided. Using: {}".format(strategy))

        if power is None and strategy == "polynomial":
            power = 2
            LOG.warning("Power value not provided. Using: {}".format(power))

        if step_size is None and strategy == "stepwise":
            step_size = 1000
            LOG.warning("Step size value not provided. Using: {}".format(step_size))

        if increment is None and strategy == "stepwise":
            increment = 0.1
            LOG.warning("Increment value not provided. Using: {}".format(increment))

        if k is None and strategy == "sigmoid":
            k = 0.1
            LOG.warning("K value not provided. Using: {}".format(k))

        self.alpha_scheduler = AlphaScheduler(
            initial_alpha=initial_alpha,
            final_alpha=final_alpha,
            max_steps=max_steps,
            strategy=strategy,
            power=power,
            step_size=step_size,
            increment=increment,
            k=k
        )
        self.alpha_scheduler_layers = self.cfg.alpha_scheduler_layers  # For layer-wise annealing
        self.comet_experiment = None
        # self.total_tokens = 0

    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        if hasattr(model, 'config'):
            model.config.adasplash_alpha = self.alpha_scheduler.initial_alpha
            model.config.alpha_scheduler_layers = self.alpha_scheduler_layers
            # self.total_tokens = 0

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Initialize Comet ML experiment reference
        if is_comet_available():
            self.comet_experiment = comet_ml.get_running_experiment()
            if self.comet_experiment is None and is_main_process():
                LOG.warning("No running Comet ML experiment found.")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        new_alpha = self.alpha_scheduler.step()

        # Update alpha in model.config (safe and model-agnostic)
        if hasattr(model, 'config'):
            model.config.adasplash_alpha = new_alpha
        elif hasattr(model, 'model') and hasattr(model.model, 'config'):
            model.model.config.adasplash_alpha = new_alpha
        else:
            LOG.warning("Model config not found; cannot update alpha dynamically.")

        # recover adasplash_effective_sequence_len
        if hasattr(model, 'config'):
            effective_sequence_len = getattr(model.config, "adasplash_effective_sequence_len", None)
        elif hasattr(model, 'model') and hasattr(model.model, 'config'):
            effective_sequence_len = getattr(model.model.config, "adasplash_effective_sequence_len", None)
        else:
            effective_sequence_len = None

        # Logging alpha value to Comet ML
        if is_main_process() and self.comet_experiment:
            self.comet_experiment.log_metric(
                "adasplash_alpha", new_alpha, step=state.global_step
            )
            # if effective_sequence_len is not None:
            #     # Logging effective sequence length to Comet ML
            #     self.comet_experiment.log_metric(
            #         "effective_sequence_len", effective_sequence_len, step=state.global_step
            #     )
            #     self.total_tokens += effective_sequence_len * args.per_device_train_batch_size
            #     self.comet_experiment.log_metric(
            #         "total_tokens", self.total_tokens, step=state.global_step
            #     )
        elif is_main_process():
            LOG.warning("Comet experiment not initialized; skipping logging.")

        return control



class AlphaScheduler:
    def __init__(
        self,
        initial_alpha=1.0000001,
        final_alpha=2.0,
        max_steps=10000,
        strategy="linear",
        power=2,  # For polynomial annealing
        step_size=1000,  # For stepwise annealing
        increment=0.1,  # For stepwise annealing
        k=0.1  # For sigmoid annealing
    ):
        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.max_steps = max_steps
        self.current_step = 0
        self.strategy = strategy
        self.power = power
        self.step_size = step_size
        self.increment = increment
        self.alpha = initial_alpha
        self.k = k

    def step(self):
        self.current_step += 1
        progress = self.current_step / self.max_steps

        if self.strategy == "linear":
            # Linear annealing
            new_alpha = self.initial_alpha + (self.final_alpha - self.initial_alpha) * progress

        elif self.strategy == "exponential":
            # Exponential annealing
            new_alpha = self.initial_alpha * (self.final_alpha / self.initial_alpha) ** progress

        elif self.strategy == "cosine":
            # Cosine annealing
            new_alpha = self.final_alpha - (self.final_alpha - self.initial_alpha) * (
                        1 + math.cos(math.pi * progress)) / 2

        elif self.strategy == "polynomial":
            # Polynomial annealing
            new_alpha = self.initial_alpha + (self.final_alpha - self.initial_alpha) * (progress ** self.power)

        elif self.strategy == "stepwise":
            # Stepwise annealing
            new_alpha = self.initial_alpha + (self.current_step // self.step_size) * self.increment
            new_alpha = min(new_alpha, self.final_alpha)

        elif self.strategy == "sigmoid":
            # Sigmoid annealing
            new_alpha = self.initial_alpha + (self.final_alpha - self.initial_alpha) / (
                        1 + math.exp(-self.k * (self.current_step - self.max_steps / 2)))

        else:
            raise ValueError(f"Unknown annealing strategy: {self.strategy}")

        # Update alpha
        self.alpha = min(new_alpha, self.final_alpha)

        # Ensure alpha does not exceed final_alpha
        return self.alpha
