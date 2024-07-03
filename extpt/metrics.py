import wandb


class WandbCtx:
    def __init__(self, project_name, run_name, config, enabled):
        self.project = project_name
        self.run_name = run_name
        self.config = config
        self.enabled = enabled

    def __enter__(self):
        if self.enabled:
            run = wandb.init(
                project=self.project,
                name=self.run_name,
                config=self.config
            )
            return run
        else:
            return None

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type or exc_value or exc_tb:
            raise
        wandb.finish()
