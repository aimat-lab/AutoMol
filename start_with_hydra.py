import hydra
from omegaconf import DictConfig
from automol.pipeline import Pipeline


@hydra.main(config_path="hydra_config", config_name="config")
def my_app(cfg: DictConfig) -> None:
    pipeline = Pipeline(cfg)
    pipeline.print_spec()


if __name__ == "__main__":
    my_app()
