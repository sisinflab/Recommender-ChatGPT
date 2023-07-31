import pandas as pd
import numpy as np
from scipy.stats import kendalltau

def calculate_jaccard_similarity(set1, set2):
    # Calculates Jaccard similarity between two sets
    intersection_size = len(set1 & set2)
    union_size = len(set1 | set2)
    return intersection_size / union_size

def compute_jaccard_similarity_for_each_user(k, top_model_file_path, rec_file_path, output_path):
    # Computes Jaccard similarity between chatGPT model and each baseline model,
    # based on the top_k recommended items for each user, and saves the results to a file

    # Read the best models
    best_models = pd.read_csv(top_model_file_path, sep="\t")

    # Read the chatGPT model and group the recommended items by user
    chatgpt_model = pd.read_csv(rec_file_path + best_models["model"].iloc[0] + ".tsv", sep='\t', header=None, names=["uid", "iid", "pred"])
    chatgpt_model_groups = chatgpt_model.groupby("uid")['iid']

    # Dictionary to store the Jaccard similarity scores for each model
    model_sim_dict = {}

    # Loop through all models except the first one (which is the baseline model)
    for n in range(1, len(best_models)):
        # Read the next model and group the recommended items by user
        model_n = pd.read_csv(rec_file_path + best_models["model"].iloc[n] + ".tsv", sep='\t', header=None, names=["uid", "iid", "pred"])
        model_n_groups = model_n.groupby("uid")['iid']

        # Calculate the Jaccard similarity score for each user between the current model and the baseline model
        avg_jaccard = []
        for user, group in chatgpt_model_groups:
            try:
                recs = group.tolist()[:k]  # Get the top 10 recommended items for the user from the baseline model
                other_recs = model_n_groups.get_group(user).tolist()[:k]  # Get the top 10 recommended items for the user from the current model
                avg_jaccard.append(calculate_jaccard_similarity(set(recs), set(other_recs)))
            except KeyError as e:
                continue

        # Calculate the average Jaccard similarity score across all users and store it in the dictionary
        model_sim_dict[best_models["model"].iloc[n]] = np.average(avg_jaccard)

    # Save the Jaccard similarity scores to a file
    result = pd.DataFrame.from_dict(model_sim_dict, orient='index', columns=['jaccard_coefficient'])
    result.index.name = 'model_name'
    result.to_csv(output_path, sep='\t')

def compute_kendall_tau_for_each_user(k, top_model_file_path, rec_file_path, output_path):
    # Computes Kendall's tau correlation coefficient between chatGPT model and each baseline model,
    # based on the top_k recommended items for each user, and saves the results to a file

    # Read the best models
    best_models = pd.read_csv(top_model_file_path, sep="\t")

    # Read the chatGPT model and group the recommended items by user
    chatgpt_model = pd.read_csv(rec_file_path + best_models["model"].iloc[0] + ".tsv", sep='\t', header=None, names=["uid", "iid", "pred"])
    chatgpt_model_groups = chatgpt_model.groupby("uid")['iid']

    # Dictionary to store the Kendall's tau scores and p-values for each model
    model_sim_dict = {}

    # Loop through all models except the first one (which is the baseline model)
    for n in range(1, len(best_models)):
        # Read the next model and group the recommended items by user
        model_n = pd.read_csv(rec_file_path + best_models["model"].iloc[n] + ".tsv", sep='\t', header=None, names=["uid", "iid", "pred"])
        model_n_groups = model_n.groupby("uid")['iid']

        # Calculate the Kendall's tau score and p-value for each user between the current model and the baseline model
        avg_kendall_tau = []
        avg_p_value = []
        for user, group in chatgpt_model_groups:
            recs = group.tolist()[:k]  # Get the top k recommended items for the user from the baseline model
            try:
                other_recs = model_n_groups.get_group(user).tolist()[:k]  # Get the top k recommended items for the user from the current model

                # Truncate the longer array to the length of the shorter array
                length = min(len(recs), len(other_recs))
                recs = recs[:length]
                other_recs = other_recs[:length]

                # Calculate the Kendall's tau score and p-value for the truncated arrays
                tau, p_value = kendalltau(recs, other_recs)
                avg_kendall_tau.append(tau)
                avg_p_value.append(p_value)
            except KeyError as e:
                continue

        avg_kendall_tau = np.array(avg_kendall_tau)
        avg_kendall_tau = np.delete(avg_kendall_tau, np.where(np.isnan(avg_kendall_tau)))
        avg_p_value = np.array(avg_p_value)
        avg_p_value = np.delete(avg_p_value, np.where(np.isnan(avg_p_value)))
        # Calculate the average Kendall's tau score and p-value across all users and store them in the dictionary
        model_sim_dict[best_models["model"].iloc[n]] = {'kendall_tau_coefficient': np.average(avg_kendall_tau), 'p_value': np.average(avg_p_value)}

    # Save the Kendall's tau scores and p-values to a file
    result = pd.DataFrame.from_dict(model_sim_dict, orient='index')
    result.index.name = 'model_name'
    result.to_csv(output_path, sep='\t')


def main():
    for k in [10, 20, 50]:
        compute_jaccard_similarity_for_each_user(k,
                                                 top_model_file_path='../../results/LLMs/MovieLens/EXP_1/performance/rec_cutoff.tsv',
                                                 rec_file_path='../../results/LLMs/MovieLens/EXP_1/recs/',
                                                 output_path='../../results/LLMs/MovieLens/EXP_1/Jaccard_Kendall/jaccard_top_'+str(k)+'.tsv')
        compute_kendall_tau_for_each_user(k,
                                          top_model_file_path='../../results/LLMs/MovieLens/EXP_1/performance/rec_cutoff.tsv',
                                          rec_file_path='../../results/LLMs/MovieLens/EXP_1/recs/',
                                          output_path='../../results/LLMs/MovieLens/EXP_1/Jaccard_Kendall/kendall_top_'+str(k)+'.tsv')

if __name__ == '__main__':
     main()
     pass