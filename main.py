import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import prepare_dataset


@hydra.main(config_path = "conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    ## 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))

    ## 2. Prepare the dataset
    trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients,cfg.batch_size, cfg.val_ratio)

    print(f"Number of clients: {cfg.num_clients}")
    print(f"Number of partitions: {len(trainloaders)}")
    print(f"Number of training batches: {len(trainloaders[0])}")
    print(f"Number of validation batches: {len(validationloaders[0])}")
    print(f"Number of test batches: {len(testloader)}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Validation ratio: {cfg.val_ratio}")

if __name__ == "__main__":
    main()