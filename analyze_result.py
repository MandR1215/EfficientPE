import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union
import glob
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

def align_test_data(log_files:List[str], output_path:str) -> None:
    global_test_bundles = set()

    for file_ind, file in enumerate(log_files):
        # read file
        with open(file, 'rb') as f:
            log = pickle.load(f)

        # extract the global test data as the global intersection
        test_bundles = set()
        bundle_to_id = dict()
        for r in log['Statistics']['test_results'].keys():
            bundles = [''.join(list(map(str, l))) for l in log['Statistics']['test_results'][r]['test_data'].tolist()]
            bundle_to_id[r] = {b:i for (i,b) in enumerate(bundles)}

            if r == 1:
                test_bundles = set(bundles)
            else:
                test_bundles = test_bundles.intersection(set(bundles))
        for r in bundle_to_id.keys():
            diff = list(set(bundle_to_id[r].keys()) - test_bundles)
            for b in diff:
                bundle_to_id[r].pop(b)
            assert set(bundle_to_id[r].keys()) == test_bundles
        
        # refine the test data by eliminating indecies with 0.0 test values
        eps = 1e-15
        for r in log['Statistics']['test_results'].keys():
            bundles, indecies = bundle_to_id[r].keys(), bundle_to_id[r].values()
            for result in log['Statistics']['test_results'][r]['result'].values():
                for i in indecies:
                    if result['targets'][i] < eps:
                        test_bundles.discard(list(bundles)[i])

        # align test data over all files
        if file_ind == 0:
            global_test_bundles = test_bundles
        else:
            global_test_bundles = global_test_bundles.intersection(test_bundles)

    with open(output_path, 'wb') as f:
        pickle.dump(global_test_bundles, f)

def test_report(log_files:List[str],
                test_bundle_path: str,
                return_values:bool=False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]:
    test_mape_records = dict()
    if return_values:
        test_data_records = dict()

    for i, file in enumerate(log_files):
        exp_name = f'exp_{i}'

        # log file
        with open(file, 'rb') as f:
            log = pickle.load(f)

        # globally aligned test data
        with open(test_bundle_path, 'rb') as f:
            test_bundles = pickle.load(f)

        # create the global test data from the global intersection
        bundle_to_id = dict()
        for r in log['Statistics']['test_results'].keys():
            bundles = [''.join(list(map(str, l))) for l in log['Statistics']['test_results'][r]['test_data'].tolist()]
            bundle_to_id[r] = {b:i for (i,b) in enumerate(bundles)}
        for r in bundle_to_id.keys():
            diff = list(set(bundle_to_id[r].keys()) - test_bundles)
            for b in diff:
                bundle_to_id[r].pop(b)
            assert set(bundle_to_id[r].keys()) == test_bundles

        # extract predictions and targets with corresponding indecies, and compute MAPE
        mape = dict()
        if return_values:
            test_data = dict()

        for r in log['Statistics']['test_results'].keys():
            indecies = bundle_to_id[r].values()
            mape[r] = list()
            if return_values:
                test_data[r] = dict()
            for bidder, result in log['Statistics']['test_results'][r]['result'].items():
                preds = [result['preds'][i] for i in indecies]
                targets = [result['targets'][i] for i in indecies]

                mape[r].append(mean_absolute_percentage_error(y_true=targets, y_pred=preds))
                if return_values:
                    test_data[r][bidder] = {'preds': preds, 'targets': targets}

        test_mape_records[exp_name] = mape
        if return_values:
            test_data_records[exp_name] = test_data
    
    # create dataframe
    df_test_mape = None
    for exp, records in test_mape_records.items():
        df_mape = pd.DataFrame.from_records(records)
        df_mape['exp'] = exp
        df_mape = df_mape.reset_index().rename(columns={'index': 'bidder'})

        if df_test_mape is None:
            df_test_mape = df_mape
        else:
            df_test_mape = pd.concat([df_test_mape, df_mape])

    if return_values:
        return df_test_mape, test_data_records
    else:
        return df_test_mape

def efficiency_report(log_files:List[str]) -> pd.DataFrame:
    test_efficiency_records = list()
    for i, file in enumerate(log_files):
        exp_name = f'exp_{i}'

        # read file
        with open(file, 'rb') as f:
            log = pickle.load(f)

        test_efficiency_records.append([exp_name, log['MLCA Efficiency']])

    df_test_efficiency = pd.DataFrame.from_records(test_efficiency_records, columns=['exp', 'efficiency'])

    return df_test_efficiency

def plot_mape_result(items:int,
                     population:List[int],
                     logs_list_normal:List[str],
                     logs_list_front:List[str],
                     logs_list_rear:List[str],
                     legend:str=None):
    try:
        os.mkdir("test_bundles")
    except FileExistsError:
        pass
    try:
        os.mkdir("mape_results")
    except FileExistsError:
        pass

    lo, re, na = population

    output_path = "test_bundles/Item{}_Qmax{}_Local{}_Regional{}_National{}.pkl".format(items, 10, lo, re, na)

    logs_list_exp = logs_list_normal + logs_list_front + logs_list_rear

    align_test_data(log_files=logs_list_exp, output_path=output_path)

    with open(output_path, 'rb') as f:
        test_set = pickle.load(f)
        print(f"{len(test_set)=}")

    # normal
    print("Normal")
    normal_mape = test_report(log_files=logs_list_normal, 
                            test_bundle_path=output_path, 
                            return_values=False).groupby('bidder').mean(numeric_only=True).mean().to_list()
    print([round(mape, 2) for mape in normal_mape], "\n")
    df_normal_mape = pd.DataFrame.from_dict({'k': list(range(1,10)), 'MAPE': normal_mape})

    # front
    print("Front")
    frontid_mape = test_report(log_files=logs_list_front,
                            test_bundle_path=output_path,
                            return_values=False).groupby('bidder').mean(numeric_only=True).mean().to_list()
    print([round(mape, 2) for mape in frontid_mape], "\n")
    df_frontid_mape = pd.DataFrame.from_dict({'k': list(range(1,10)), 'MAPE': frontid_mape})



    # rear
    print("Rear")
    rearid_mape = test_report(log_files=logs_list_rear,
                            test_bundle_path=output_path,
                            return_values=False).groupby('bidder').mean(numeric_only=True).mean().to_list()
    print([round(mape, 2) for mape in frontid_mape], "\n")
    df_rearid_mape = pd.DataFrame.from_dict({'k': list(range(1,10)), 'MAPE': rearid_mape})

    # plot curves
    plt.rcParams["font.size"] = fontsize = 23
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix' 

    plt.figure(figsize=(6,4))
    linewidth = 3
    markersize = 12
    sns.lineplot(data=df_normal_mape, 
                x='k', 
                y='MAPE', 
                linestyle='solid',
                marker='o',
                markersize=markersize,
                linewidth=linewidth,
                label='MVNN (baseline)')

    sns.lineplot(data=df_frontid_mape,
                x='k',
                y='MAPE',
                linestyle='dashed',
                marker='*',
                markersize=markersize+3,
                linewidth=linewidth,
                label='MT-MLCA-F (ours)')

    sns.lineplot(data=df_rearid_mape,
                x='k',
                y='MAPE',
                linestyle='dashdot',
                marker='^',
                markersize=markersize,
                linewidth=linewidth,
                label='MT-MLCA-R (ours)')
    plt.title('{} Items, (Lo, Re, Na) = ({},{},{})'.format(items, lo, re, na))
    plt.grid(color = "gray", linewidth=1, alpha=.5)
    plt.tick_params(axis='x', labelsize=fontsize+5)
    plt.tick_params(axis='y', labelsize=fontsize+3)
    plt.xlabel(r'$k$', fontsize=fontsize+7)
    if not legend:
        plt.legend().set_visible(False)
    else:
        plt.legend(loc=legend, fontsize=fontsize-5)
    plt.ylabel('MAPE', fontsize=fontsize+5)
    plt.yticks(np.arange(0.5, 0.710, 0.02))
    plt.ylim([0.565, 0.705])
    plt.xticks(np.arange(1,10,1))
    plt.savefig("mape_results/mape_result_item{}_lo{}_re{}_na{}.pdf".format(items, lo, re, na), 
                transparent=True,
                bbox_inches='tight')
    plt.show()

    
def plot_efficiency_result(items:int,
                           logs_eff_normal:List[str],
                           logs_eff_front:List[str],
                           logs_eff_rear:List[str],
                           legend:str=None):
    try:
        os.mkdir("efficiency_results")
    except FileExistsError:
        pass

    # efficiency results
    efficienciy_results = []
    for logs_eff in [logs_eff_normal, logs_eff_front, logs_eff_rear]:
        reports = []
        for i, log in enumerate(logs_eff):
            df_report = efficiency_report(log).drop(columns='exp')
            df_report['(Local, Regional, National)'] = str((3*(i+1), 4*(i+1), 3*(i+1)))
            reports.append(df_report)
        efficienciy_results.append(pd.concat(reports, axis=0).reset_index(drop=True))

    df_eff_normal, df_eff_front, df_eff_rear = efficienciy_results

    # plot
    plt.rcParams["font.size"] = fontsize = 23
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix' 

    plt.figure(figsize=(8,6))
    linewidth = 3
    markersize = 12
    sns.lineplot(data=df_eff_normal, 
                 x='(Local, Regional, National)', 
                 y='efficiency',
                 err_style=None, 
                 marker='o',
                 markersize=markersize,
                 linewidth=linewidth,
                 label='MVNN (baseline)',
                 linestyle='solid')
    sns.lineplot(data=df_eff_front, 
                 x='(Local, Regional, National)', 
                 y='efficiency',
                 err_style=None, 
                 marker='*', 
                 markersize=markersize+3,
                 linewidth=linewidth,
                 label='MT-MLCA-F (ours)',
                 linestyle='dashed')
    sns.lineplot(data=df_eff_rear, 
                 x='(Local, Regional, National)', 
                 y='efficiency',
                 err_style=None, 
                 marker='^',
                 markersize=markersize,
                 linewidth=linewidth,
                 label='MT-MLCA-R (ours)',
                 linestyle='dashdot')
    if not legend:
        plt.legend().set_visible(False)
    else:
        plt.legend(loc=legend)
    plt.xlabel('Population')
    plt.ylabel('Efficiency')
    plt.title(f'{items} Items')
    plt.grid(color = "gray", linewidth=1, alpha=.5)
    plt.tick_params(axis='x', labelsize=fontsize-3)
    plt.yticks(np.arange(0.40, 0.6, 0.025))
    plt.ylim([0.39, 0.56])
    plt.tight_layout()
    plt.savefig(f"efficiency_results/efficiency_result_{items}.pdf",
                transparent=True)
    plt.show()