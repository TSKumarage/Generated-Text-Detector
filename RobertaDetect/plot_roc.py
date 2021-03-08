#!/usr/bin/env python3

'''
This script can use the analytic output scored against truth data, e.g. Dry-Run_Data from TA3,
and generate an image of the ROC curve written to a file.

For a single analtyic with the detection evaluation task:

    usage: python plot_roc.py -o ~/roc-plot.pdf -t DETECTION -i
               Dry-Run_Data/Primary_Tasks/Results/purdue-unisi-gandetection_results.json

Or for plotting multiple analytics at once for the (default) ATTRIBUTION task:

usage:
    python plot_roc.py -i Dry-Run_Data/Primary_Tasks/Results/*.json
    python plot_roc.py -o ~/roc-plot.pdf -i Dry-Run_Data/Primary_Tasks/Results/*.json

TODO: add version when available
TODO: plot metrics and thresholdedMetrics on the ROC plots

'''

import argparse
import yaml
import sys
import glob
import matplotlib.pyplot as plt
from matplotlib.rcsetup import cycler
from sklearn.metrics import roc_curve, auc
#import pprint

def main(args):
    print(f"Generating ROC with these arugments: {args}")

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    #linestyles = ['-', '--', ':', '-.']
    lw=3

    plt.figure()
    print("Printing stats in CSV format:")
    # Make sure this CSV list is in sync with str_print below
    print("AnalyticName,EER,AUC,N-scored,N-Targets,N-optOut,N-errors,N-total,PD-thresh,FAR-thresh")
    for i, fname in enumerate(args.file_paths):
        with open(fname) as f:
            J=yaml.safe_load(f)

        #print(f"Processing: {fname}")
        # pp=pprint.PrettyPrinter(indent=1,width=60,compact=True, depth=3)
        # pp.pprint(J)

        analyticName = J['analyticName']
        isTargets = []
        probDets = []
        llrs = []
        probeIds = []
        probFas = []

        for result in J['results']:
            if result['taskName'] == args.task_name:
                for score in result['scores']:
                    isTargets.append(score['isTarget'])
                    probDets.append(score['probDet'])
                    llrs.append(score['llr'])
                    probeIds.append(score['probeId'])
                    probFas.append(score['probFa'])

                total_probes = len(result['errors']) + len(result['optOut']) + len(isTargets)
                eer = result['metrics'][1]['value'] if result['metrics'][1]['value'] else 0.

                if probFas and probDets:
                    roc_auc = auc(probFas, probDets)
                else:
                    roc_auc = 0.

                str_key="%s:\n %0.2f EER, %0.2f AUC, %d scored (%d T), %d optOut, %d err"\
                    %(analyticName, eer, roc_auc, len(isTargets), sum(isTargets),\
                    len(result['optOut']), len(result['errors']))

                col=colors[i % len(colors)]
                #sty=linestyles[i % len(linestyles)]
                if probFas and  probDets:
                    plt.plot(probFas, probDets, label=str_key, color=col, lw=lw) # linestyle=sty,

                for metric in result['thresholdedMetrics'][0]['metrics']:
                    if metric['metricName'] == 'Probability of Detection':
                        pd=metric['value']

                    if metric['metricName'] == 'False Alarm Rate':
                        far=metric['value']

                if pd and far:
                    plt.plot(far, pd, markersize=12, marker='o', color=col)
                else:
                    pd=0.
                    far=0.

                str_print=f"{analyticName},{eer:.2f},{roc_auc:.2f},{len(isTargets)},"\
                    f"{sum(isTargets)},{len(result['optOut'])},{len(result['errors'])},"\
                    f"{total_probes},{pd:.2f},{far:.2f}"
                print(str_print)

    #plt.rc('lines', linewidth=lw)
    #plt.rc('axes', prop_cycle=(cycler('color', ['b', 'g', 'r', 'c', 'm', 'y', 'k'])+
    #                         cycler('linestyle', ['-', '--', ':', '-.'])))
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Probability of False Alarm')
    plt.ylabel('Probability of Detection')
    plt.legend(loc="lower right")
    plt.title(f"Task: {args.task_name}")
    plt.savefig(args.roc)
    print(f"ROC rendered to file: {args.roc}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Write the ROC plot to a file given one or more JSON scoring files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--roc', '-o', dest='roc', type=str, default='roc.pdf',
        help='path to the (pdf/png) image file for the ROC.')
    parser.add_argument('--task-name', '-t', dest='task_name', type=str, default='DETECTION',
        help='DETECTION, ATTRIBUTION, or CHARACTERIZATION evaluation task.')
    parser.add_argument('--json', '-i', dest='file_paths', type=str, nargs='+',
        required=True, help='One or more JSON files with scores as input. \n'
        'Also accepts glob like foo/*.json.', default='/content/drive/Shareddrives/DARPA/Datasets/Eval1Sources/kitware-asu-generated-text-detection_results.json')
    sys.exit(main(parser.parse_args()))

