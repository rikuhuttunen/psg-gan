"""Create pretty confusion matrix plots for the paper."""
import argparse
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from psg_gan.metrics import classification


# Hard-code here, since the conf mats were only saved as PDF plots during training
# mlflow run name: merciful-donkey-886
REAL_CONF_MATS = {
    'MESA': {
        'n': 100,
        'hg_kappa': 0.72,
        'events_kappa': 0.59,
        'hg': [
            [20612, 2252, 399, 2, 467],
            [1461, 4041, 3719, 9, 742],
            [1449, 1996, 34522, 2879, 940],
            [52, 5, 1831, 5266, 2],
            [400, 120, 152, 1, 12267]
        ],
        'events': [
            [2500954, 53586, 19413],
            [100826, 108595, 22482],
            [6119, 7340, 48265]
        ]
    },
    # 'MSTR': {
    #     'n': 100,
    #     'hg_kappa': 0.69,
    #     'events_kappa': 0.60,
    #     'hg': [
    #         [12062, 493, 1007, 35, 626],
    #         [1729, 2442, 3130, 5, 1193],
    #         [623, 1507, 31451, 3470, 1229],
    #         [13, 1, 1091, 6526, 0],
    #         [183, 95, 344, 0, 9306]
    #     ],
    #     'events': [
    #         [2060980, 57356, 13272],
    #         [69908, 96123, 19071],
    #         [2670, 5559, 33393]
    #     ]
    # },
    'STAR-SA': {
        'n': 50,
        'hg_kappa': 0.67,
        'events_kappa': 0.63,
        'hg': [
            [10045, 1324, 973, 5, 410],
            [978, 1894, 2609, 7, 559],
            [624, 774, 15586, 918, 444],
            [48, 2, 1414, 4065, 7],
            [315, 103, 301, 0, 5608]
        ],
        'events': [
            [1194138, 42053, 9955],
            [53642, 67578, 13219],
            [7119, 17324, 65362]
        ]
    },
    'STAR-HV': {
        'n': 100,
        'hg_kappa': 0.76,
        'events_kappa': 0.56,
        'hg': [
            [20123, 2828, 2668, 51, 1388],
            [205, 955, 1693, 2, 184],
            [1096, 1151, 36607, 3026, 991],
            [70, 4, 1115, 12275, 6],
            [447, 127, 572, 6, 14364]
        ],
        'events': [
            [2904676, 44826, 2824],
            [30063, 48782, 2043],
            [4244, 11507, 9655]
        ]
    }
}

SYNTHETIC_CONF_MATS = {
    'MESA': {
        'n': 100,
        'hg_kappa': 0.64,
        'events_kappa': 0.52,
        'hg': [
            [19894, 2318, 507, 9, 1004],
            [2268, 3124, 3550, 30, 1000],
            [3252, 2867, 30654, 3656, 1357],
            [141, 26, 1681, 5287, 21],
            [713, 145, 196, 4, 11882]
        ],
        'events': [
            [2505488, 46892, 21573],
            [121418, 78042, 32443],
            [7682, 7462, 46580]
        ]
    },
    # 'MSTR': {
    #     'n': 100,
    #     'hg_kappa': 0.69,
    #     'events_kappa': 0.60,
    #     'hg': [
    #         [12062, 493, 1007, 35, 626],
    #         [1729, 2442, 3130, 5, 1193],
    #         [623, 1507, 31451, 3470, 1229],
    #         [13, 1, 1091, 6526, 0],
    #         [183, 95, 344, 0, 9306]
    #     ],
    #     'events': [
    #         [2060980, 57356, 13272],
    #         [69908, 96123, 19071],
    #         [2670, 5559, 33393]
    #     ]
    # },
    'STAR-SA': {
        'n': 50,
        'hg_kappa': 0.63,
        'events_kappa': 0.60,
        'hg': [
            [10162, 1293, 873, 4, 425],
            [1389, 1469, 2526, 14, 649],
            [1132, 987, 13766, 1933, 528],
            [102, 6, 752, 4669, 7],
            [398, 74, 282, 3, 5570]
        ],
        'events': [
            [1212360, 24662, 9124],
            [68055, 53276, 13108],
            [7879, 22537, 59389]
        ]
    },
    'STAR-HV': {
        'n': 100,
        'hg_kappa': 0.71,
        'events_kappa': 0.57,
        'hg': [
            [20278, 2848, 2237, 93, 1602],
            [386, 701, 1669, 13, 270],
            [2108, 1473, 32776, 5466, 1048],
            [223, 12, 488, 12740, 7],
            [705, 106, 587, 6, 14112]
        ],
        'events': [
            [2927562, 22095, 2669],
            [38384, 38926, 3578],
            [4557, 9573, 11276]
        ]
    }
}

HG_CLASS_NAMES = ['W', 'N1', 'N2', 'N3', 'REM']
EVENT_CLASS_NAMES = ['No event', 'Hypopnea', 'Apnea']


def plot_confmats(
        confmats,
        hg_figsize=(5, 5),
        event_figsize=(4.5, 4.5),
        xlabel='Model',
        ylabel='Ground truth'):
    hg_fig, hg_axs = plt.subplots(
         1, 3,
         figsize=(0.91 * 3* hg_figsize[0], 0.97 * hg_figsize[1]),
         layout='constrained')
    event_fig, event_axs = plt.subplots(
         1, 3,
         figsize=(0.85 * 3 * event_figsize[0], 0.95 * event_figsize[1]),
         layout='constrained')

    for i, dataset in enumerate(confmats):
            # row = i // 2
            # col = i % 2
            # show_yticks = col == 0
            # show_xticks = row == 1
            show_yticks = i == 0
            _ = classification.plot_cm(
                 np.array(confmats[dataset]['hg']),
                 class_names=HG_CLASS_NAMES,
                 fontsize=14,
                 #ax=hg_axs[row, col],
                 ax=hg_axs[i],
                 show_yticks=show_yticks,
                 #show_xticks=show_xticks,
                 #xlabel='Automatic' if show_xticks else None,
                 xlabel=xlabel,
                 ylabel=ylabel if show_yticks else None,
            )
            _ = classification.plot_cm(
                 np.array(confmats[dataset]['events']),
                 class_names=EVENT_CLASS_NAMES,
                 fontsize=14,
                 #ax=event_axs[row, col],
                 ax=event_axs[i],
                 show_yticks=show_yticks,
                 #show_xticks=show_xticks,
                 #xlabel='Automatic' if show_xticks else None,
                 xlabel=xlabel,
                 ylabel=ylabel if show_yticks else None,
            )
            #hg_axs[row, col].set_title(
            hg_axs[i].set_title(
                 fr'{dataset}, $n = {confmats[dataset]['n']}$, $\kappa = {confmats[dataset]['hg_kappa']}$',
                 fontsize=18)
            #event_axs[row, col].set_title(
            event_axs[i].set_title(
                 fr'{dataset}, $n = {confmats[dataset]['n']}$, $\kappa = {confmats[dataset]['events_kappa']}$',
                 fontsize=18)

    return hg_fig, hg_axs, event_fig, event_axs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', type=Path, required=True)

    args = parser.parse_args()
    real_hg_fig, real_hg_axs, real_event_fig, real_event_axs = plot_confmats(REAL_CONF_MATS, xlabel='Real model')
    fake_hg_fig, fake_hg_axs, fake_event_fig, fake_event_axs = plot_confmats(SYNTHETIC_CONF_MATS, xlabel='Synthetic model')

    args.save_dir.mkdir(exist_ok=True)
    real_hg_fig.savefig(args.save_dir / 'real_hg_confmats.pdf', dpi=500, bbox_inches='tight')
    real_event_fig.savefig(args.save_dir / 'real_event_confmats.pdf', dpi=500, bbox_inches='tight')

    fake_hg_fig.savefig(args.save_dir / 'synthetic_hg_confmats.pdf', dpi=500, bbox_inches='tight')
    fake_event_fig.savefig(args.save_dir / 'synthetic_event_confmats.pdf', dpi=500, bbox_inches='tight')
