import pandas as pd
import re

def prase_exp_name(exp_name):
    # food-syn_F_lambda0.001_gamma0.2_model-resnet50
    exp_name_split = exp_name.split("_")
    # use regex to extract lambda and gamma
    lambda_value = re.search(r"lambda([.\d]+)", exp_name)
    lambda_value = lambda_value.group(1) if lambda_value else -1
    gamma_value = re.search(r"gamma([.\d]+)", exp_name)
    gamma_value = gamma_value.group(1) if gamma_value else -1
    model_name = re.search(r"model-(\w+)", exp_name)
    model_name = model_name.group(1) if model_name else None
    if model_name is None:
        if exp_name_split[0].split('-')[0] == "cifar":
            model_name = "resnet9"
        elif exp_name_split[0].split('-')[0] == "music":
            model_name = "fcnn_dropout_batchnorm"
    details = {
        "dataset": exp_name_split[0],
        "crowd_config": exp_name_split[1],
        "lambda": lambda_value,
        "gamma": gamma_value,
        "model": model_name
    }
    return details


def main():
    df = pd.read_csv("logs/results.txt", names=["exp_name", "seed", "test_accuracy"])
    df["seed"] = df["seed"].astype(int)

    # parse experiment name
    df["exp_details"] = df["exp_name"].apply(prase_exp_name)
    df = df.join(pd.json_normalize(df["exp_details"]))

    # remove duplicates
    print("Duplicates: ", df[df.duplicated(subset=["exp_name", "seed"])])
    print("Experiments Count", len(df))
    unique_df = df.drop_duplicates(subset=["exp_name", "seed"])
    print("Unique experiments: ", len(unique_df))

    # aggregate different seeds and calculate mean and std
    unique_df = unique_df.groupby(["dataset", "crowd_config", "lambda", "gamma", "model"]).agg(
        mean_accuracy=("test_accuracy", "mean"),
        std_accuracy=("test_accuracy", "std"),
    ).reset_index()

    unique_df.to_csv("logs/results.csv", index=False)


if __name__ == '__main__':
    main()
