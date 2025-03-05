from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/cx/cx1/github-repo/SeqTrackv2/data/got10k_lmdb'
    settings.got10k_path = '/home/cx/cx1/github-repo/SeqTrackv2/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = '/home/cx/cx1/github-repo/SeqTrackv2/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/cx/cx1/github-repo/SeqTrackv2/data/lasot_lmdb'
    settings.lasot_path = '/home/cx/cx1/github-repo/SeqTrackv2/data/lasot'
    settings.lasotlang_path = '/home/cx/cx1/github-repo/SeqTrackv2/data/lasot'
    settings.network_path = '/home/cscv/Documents/lsl/SeqTrackv2/lib/train/checkpoints/train/seqtrackv2/seqtrackv2_b256'    # Where tracking networks are stored.
    settings.nfs_path = '/home/cx/cx1/github-repo/SeqTrackv2/data/nfs'
    settings.otb_path = '/home/cx/cx1/github-repo/SeqTrackv2/data/OTB2015'
    settings.otblang_path = '/home/cx/cx1/github-repo/SeqTrackv2/data/otb_lang'
    settings.prj_dir = '/home/cscv/Documents/lsl/SeqTrackv2'
    settings.result_plot_path = '/home/cx/cx1/github-repo/SeqTrackv2/test/result_plots'
    settings.results_path = '/home/cscv/Documents/lsl/SeqTrackv2/tracking/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/cscv/Documents/lsl/SeqTrackv2/lib/train'
    settings.segmentation_path = '/home/cx/cx1/github-repo/SeqTrackv2/test/segmentation_results'
    settings.tc128_path = '/home/cx/cx1/github-repo/SeqTrackv2/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/cx/cx1/github-repo/SeqTrackv2/data/tnl2k/test'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/cx/cx1/github-repo/SeqTrackv2/data/trackingnet'
    settings.uav_path = '/home/cx/cx1/github-repo/SeqTrackv2/data/UAV123'
    settings.vot_path = '/home/cx/cx1/github-repo/SeqTrackv2/data/VOT2019'
    settings.youtubevos_dir = ''
    settings.gtot_path = '/home/cscv/Documents/lsl/dataset/GTOT'
    settings.lasher_path = '/media/cscv/d00985a0-c3e6-4ffa-9546-88c861db5ce3/02_Dataset/LasHeR'
    settings.rgbt234_path = '/home/cscv/Documents/lsl/dataset/RGB-T234'
    settings.vtuav_path = '/media/cscv/d00985a0-c3e6-4ffa-9546-88c861db5ce3/02_Dataset/VTUAV'

    return settings

