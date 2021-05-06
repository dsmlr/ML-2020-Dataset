from zipfile import ZipFile
from glob import glob
from purity import purity_score
from datetime import datetime

import pandas as pd
import numpy as np

import os
import shutil

if __name__ == "__main__":
    rank_dict = {
        'name': [],
        'filename': [],
        'score': [],
        'note': []
    }

    unzip_dir = 'submissions/outputs'
    true_df = pd.read_csv('Stars Clustering/Stars_original.csv')
    y_true = true_df['Type'].to_numpy()

    try:
        shutil.rmtree(unzip_dir)
    except FileNotFoundError:
        print(f'No {unzip_dir} directory. Creating new one..')

    zip_filename = 'submissions/ML-2-20-Submission - Predicted Label for ClusteringTask-7311.zip'
    with ZipFile(zip_filename, 'r') as zipObj:
        zipObj.extractall(unzip_dir)

    outputs_list = glob(unzip_dir+'/*')
    for output_dir in outputs_list:
        name_splitext = os.path.splitext(os.path.basename(output_dir))[0]
        name = name_splitext.split('_')[0]
        # print(name)
        answer_files = glob(output_dir+'/*.csv')

        for answer_file in answer_files:
            answer_df = pd.read_csv(answer_file)
            last_column = answer_df.columns[-1]
            answer_df = answer_df.sort_values(by=[last_column])

            # print(answer_df[answer_df.columns[-1]])
            filename = os.path.splitext(os.path.basename(answer_file))[0]
            filename = filename+".csv"

            # print(filename)
            y_pred = answer_df[answer_df.columns[-1]].to_numpy()

            try:
                score = purity_score(y_true, y_pred)
                rank_dict['name'].append(name)
                rank_dict['filename'].append(filename)
                rank_dict['score'].append(score)
                rank_dict['note'].append('')
            except:
                rank_dict['name'].append(name)
                rank_dict['filename'].append(filename)
                rank_dict['score'].append(0)
                rank_dict['note'].append('Error')

    rank_df = pd.DataFrame(rank_dict)
    rank_df = rank_df.sort_values(by=['score'], ascending=False)
    rank_df.index = np.arange(1, len(rank_df) + 1)
    
    html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Star Clustering Leaderboard</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <h1 class="topic">Star Clustering Leaderboard</h1>
    <p>Last updated: {update_datetime}</p>
    {table}
</body>
</html>
    '''

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    html_source = html_template.format(
        update_datetime=dt_string, 
        table=rank_df.to_html(classes='leaderboard', justify='left')
    )

    f = open("docs/index.html", "w")
    f.write(html_source)
    f.close()