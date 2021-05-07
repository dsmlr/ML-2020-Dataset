from zipfile import ZipFile
from glob import glob
from purity import purity_score
from datetime import datetime
from sklearn.metrics import normalized_mutual_info_score, pairwise_distances
from sklearn.metrics.cluster import rand_score

import pandas as pd
import numpy as np

import os
import shutil

if __name__ == "__main__":
    rank_dict = {
        'name': [],
        'filename': [],
        'purity': [],
        'nmi': [],
        'rand_index': [],
        'score': [],
        'note': []
    }

    unzip_dir = '/home/phongsathorn/Projects/ML-2020-Dataset/submissions/outputs'
    true_df = pd.read_csv('/home/phongsathorn/Projects/ML-2020-Dataset/Stars Clustering/Stars answer.csv')

    y_true = true_df['Type'].to_numpy()

    try:
        shutil.rmtree(unzip_dir)
    except FileNotFoundError:
        print(f'No {unzip_dir} directory. Creating new one..')

    zip_filename = '/home/phongsathorn/Projects/ML-2020-Dataset/submissions/submission.zip'
    with ZipFile(zip_filename, 'r') as zipObj:
        zipObj.extractall(unzip_dir)

    outputs_list = glob(unzip_dir+'/*')
    for output_dir in outputs_list:
        name_splitext = os.path.splitext(os.path.basename(output_dir))[0]
        name = name_splitext.split('_')[0]
        answer_files = glob(output_dir+'/*.csv')

        for answer_file in answer_files:
            answer_df = pd.read_csv(answer_file)
            last_column = answer_df.columns[-1]
            answer_df = answer_df.sort_values(by=[last_column])

            filename = os.path.splitext(os.path.basename(answer_file))[0]
            filename = filename+".csv"

            y_pred = answer_df[answer_df.columns[-1]].to_numpy()

            try:
                purity = purity_score(y_true, y_pred)
                nmi = normalized_mutual_info_score(y_true, y_pred)
                rand = rand_score(y_true, y_pred)

                avg = (purity + nmi + rand) / 3

                rank_dict['name'].append(name)
                rank_dict['filename'].append(filename)
                rank_dict['purity'].append(purity)
                rank_dict['nmi'].append(nmi)
                rank_dict['rand_index'].append(rand)
                rank_dict['score'].append(avg)
                rank_dict['note'].append('')
            except:
                rank_dict['name'].append(name)
                rank_dict['filename'].append(filename)
                rank_dict['purity'].append(0)
                rank_dict['nmi'].append(0)
                rank_dict['rand_index'].append(0)
                rank_dict['score'].append(0)
                rank_dict['note'].append('Error')

    rank_df = pd.DataFrame(rank_dict)
    rank_df = rank_df.sort_values(by=['score'], ascending=False)
    
    rank_df.index = np.arange(1, len(rank_df) + 1)
    nodup_rank_df = rank_df.drop_duplicates(subset=['name'])
    
    print(rank_df)
    
    html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Star Clustering Leaderboard</title>
    <link rel="stylesheet" href="style.css">
    <script src="script.js"></script>
</head>
<body>
    <h1 class="topic">✨ Star Clustering Leaderboard</h1>
    <p>🕓 Lastest update: {update_datetime}</p>
    <p>(Update every 30 minutes, up to 5 files per submission)</p>

    <div class="tabs-bar">
      <button onclick="opentab('all')">All</button>
      <button onclick="opentab('max')">Ranking</button>
    </div>

    <div id="all" class="tab" style="display:block">
    {table_all}
    </div>
    <div id="max" class="tab" style="display:none">
    {table_rank}
    </div>

    <div class="learnmore">
    <p>
        The evaluation metric for this task is purity metric, normalized mutual information (NMI) and rand index.<br>
        The score calculates by average all values from 3 metrics.<br>
        <u>Learn more:</u>
    </p>
    <ul>
        <li><a href="https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html">https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html</a></li>
        <li><a href="https://gist.github.com/jhumigas/010473a456462106a3720ca953b2c4e2">https://gist.github.com/jhumigas/010473a456462106a3720ca953b2c4e2</a></li>
        <li><a href="https://towardsdatascience.com/evaluation-metrics-for-clustering-models-5dde821dd6cd">https://towardsdatascience.com/evaluation-metrics-for-clustering-models-5dde821dd6cd</a></li>
        <li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html">https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html</a></li>
        <li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.rand_score.html#sklearn.metrics.rand_score">https://scikit-learn.org/stable/modules/generated/sklearn.metrics.rand_score.html#sklearn.metrics.rand_score</a></li>
    </ul>
    </div>
</body>
</html>
    '''

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    html_source = html_template.format(
        update_datetime=dt_string, 
        table_all=rank_df.to_html(classes='leaderboard', justify='left'),
        table_rank=nodup_rank_df.to_html(classes='leaderboard', justify='left')
    )

    f = open("/home/phongsathorn/Projects/ML-2020-Dataset/docs/index.html", "w")
    f.write(html_source)
    f.close()

    print("Generated HTML leaderboard.")
