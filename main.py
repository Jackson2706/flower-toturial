import hydra
from omegaconf import DictConfig, OmegaConf
import flwr as fl
from dataset import prepare_dataset
from client import generate_client_fn

@hydra.main(config_path = "conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    ## 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))

    ## 2. Prepare the dataset
    trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients,cfg.batch_size, cfg.val_ratio)

    # print(f"Number of clients: {cfg.num_clients}")
    # print(f"Number of partitions: {len(trainloaders)}")
    # print(f"Number of training batches: {len(trainloaders[0])}")
    # print(f"Number of validation batches: {len(validationloaders[0])}")
    # print(f"Number of test batches: {len(testloader)}")
    # print(f"Batch size: {cfg.batch_size}")
    # print(f"Validation ratio: {cfg.val_ratio}")

    # 3. Define the Flower client
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)

    # 4. Define the strategu
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.00001,
        min_fit_clients=cfg.num_clients_per_round_fit,
        fraction_evaluate=0.00001,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=
    )

if __name__ == "__main__":
    main()