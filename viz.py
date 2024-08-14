import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import json
import os
import glob

# Path to the directory containing JSON results
folder_path = './results_needle/results'  # Replace with your folder path

model_names = glob.glob(f"{folder_path}/*")
# trim all names
# split by /
model_names = [name.split("/")[-1] for name in model_names]

pick = [f"len_{step}" for step in range(1000, 226280, 4096)]
for model_name in model_names:
    # Using glob to find all json files in the directory
    json_files = glob.glob(f"{folder_path}/{model_name}/*.json")
    print(json_files)
    # List to hold the data
    data = []

    # Iterating through each file and extract the 3 columns we need
    cnt = 0
    score_sum = 0
    for file in json_files:
        name = file.split("/")[-1]
        short = name[name.find("len_"):name.find("_depth")]
        if short not in pick:
            continue

        # print(short)
        # input()
        with open(file, 'r') as f:
            json_data = json.load(f)
            # Extracting the required fields
            document_depth = json_data.get("depth_percent", None)
            context_length = json_data.get("context_length", None)
            score = json_data.get("score", None)
            # Appending to the list
            data.append({
                "Document Depth": document_depth,
                "Context Length": context_length,
                "Score": score
            })
            score_sum += score
            cnt += 1


    # Creating a DataFrame
    df = pd.DataFrame(data)

    print (df.head())
    print (f"You have {len(df)} rows")

    pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score") # This will turn into a proper pivot
    pivot_table.iloc[:5, :5]

    # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    # Create the heatmap with better aesthetics
    plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    sns.heatmap(
        pivot_table,
        # annot=True,
        vmax=10,
        vmin=0,
        fmt="g",
        cmap=cmap,
        linewidths=1,
        cbar_kws={'label': 'Score'}
    )

    # More aesthetics
    plt.title(f'Pressure Testing {model_name} Context\nFact Retrieval Across Context Lengths ("Needle In A HayStack")\n Mean Score: {score_sum/cnt}')  # Adds a title
    plt.xlabel('Token Limit')  # X-axis label
    plt.ylabel('Depth Percent')  # Y-axis label
    plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area

    # Show the plot
    plt.savefig(f'{model_name}.png')  # Save the plot as an image
    # plt.savefig(f'{model_name}.pdf', format="pdf")  # Save the plot as an image
    # plt.show()


