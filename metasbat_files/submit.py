import pandas as pd
import os

# TODO: use .txt file instead of .csv file?


# convert dictionay to cmd args in the form "--key value"
def dict_to_cmd_args(d):
    s = []
    for key, value in d.items():
        s.append("--"+key+" "+str(value))
    return " ".join(s)


def read_job_id_from_job_name(job_name):
    # find slurm job id based on given job name and read it
    job_id = os.popen('squeue --noheader --format %i --name {}'
                      .format(job_name)).read()
    # since they are all array jobs, only take the job id not the array id
    dependency_job_id = job_id.split("_")[0]
    return dependency_job_id


def main():
    # create jobs file text in this script from which a temporary file will
    # be created
    job_file = (
        "#!/bin/bash\n"
        "# redirect the output/error to some files\n"
        "#SBATCH -o /home/fiederer/out/%A-%a.o\n"
        "#SBATCH -e /home/fiederer/error/%A-%a.e\n"
        # "export PYTHONPATH={}\n"
        "{}\n"
        "{}\n"
        "python {} {}\n")

    configs_file = "/home/fiederer/nicebot/metasbat_files/configs.csv"
    # load all the configs to be run
    configs_df = pd.DataFrame.from_csv(configs_file).T

    # specify python path, virtual env and python script to be run
    # python_path = '/home/fiederer/nicebot'
    source_call = 'source /home/fiederer/.bashrc'
    virtual_env = 'conda activate braindecode'
    python_file = ('/home/fiederer/nicebot/deepRegressionCode/DeepRegression_withinSubjects_kisbat.py')

    # specify temporary job file and command to submit
    # schedule to different hosts. only one jost per host
    # hosts = ["metagpua", "metagpub", "metagpuc", "metagpud", "metagpue"]
    # queue = "ml_gpu-rtx2080"
    script_name = "/home/fiederer/jobs/slurm/run_tmp.sh"
    # batch_submit = "sbatch -p queue -w host -c num_workers --array={}-{} --job-name=b_{}_j_{} {} {}"

    # sbatch -p meta_gpu-ti -w metagpub -c 4 jobs/slurmbjob.pbs

    # loop through all the configs
    for setting in configs_df:
        model_name = configs_df[setting]['model_name']
        if any([model_name == a for a in ['lin_reg', 'lin_svr', 'rbf_svr', 'rf_reg']]):
            # Specify queue
            # queue = 'ml_cpu-ivy'
            queue = 'cpu_ivy'
            if model_name == 'rf_reg':
                batch_submit = "sbatch -p {queue} -c 16 {script_name}"
            else:
                batch_submit = "sbatch -p {queue} -c 1 {script_name}"
        elif any([model_name == a for a in ['eegnet', 'deep4', 'resnet']]):
            queue = "meta_gpu-black"
            # queue = "ml_gpu-rtx2080"
            batch_submit = "sbatch -p {queue} -c 2 --gres=gpu:1 {script_name}"
        else:
            os.warnin('Cannot define queue for model {:s}'.format(model_name))

        config = configs_df[setting].to_dict()
        # create a tmp job file / job for every config
        cmd_args = dict_to_cmd_args(config)
        curr_job_file = job_file.format(source_call,
                                        virtual_env,
                                        python_file, cmd_args)

        # write tmp job file and submit it to slurm
        with open(script_name, "w") as f:
                f.writelines(curr_job_file)

        # print(batch_submit.format(queue=queue, script_name=script_name))
        os.system(batch_submit.format(queue=queue, script_name=script_name))


if __name__ == '__main__':
    # TODO: add arg parse
    main()
