import flwr as fl
import hydra
from omegaconf import DictConfig, OmegaConf

from client import generate_client_fn
from dataset import prepare_dataset
from server import get_evaluate_fn, get_in_fit_config


@hydra.main(config_path = "conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    ## 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))

    ## 2. Prepare the dataset
    trainloaders, validationloaders, testloader = prepare_dataset(
        cfg.num_clients,cfg.batch_size, cfg.val_ratio
    )

    # 3. Define the Flower client
    client_fn = generate_client_fn(
        trainloaders, validationloaders, cfg.num_classes
    )

    # 4. Define the strategu
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.00001,
        min_fit_clients=cfg.num_clients_per_round_fit,
        fraction_evaluate=0.00001,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=get_in_fit_config(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
    )

    ### 5. Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config = fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy
    )

    ### 6. Save the results
    

if __name__ == "__main__":
    main()