import argparse, subprocess, sys


def run(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.run([sys.executable, "-m"] + cmd, check=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--max-train",type=int,default=None)
    p.add_argument("--max-localization",type=int,default=100)
    a=p.parse_args()

    run(["src.prepare_dataset"])
    train=["src.train_classifier","--epochs",str(a.epochs)]
    if a.max_train: train += ["--max-train",str(a.max_train)]
    run(train)
    run(["src.evaluate_classifier"])
    run(["src.baselines","--max-train","3000","--max-test","1000"])
    run(["src.run_localization","--max-samples",str(a.max_localization)])
    run(["src.threshold_experiment","--max-samples",str(a.max_localization)])
    print("\nPipeline completed. Check results/.")


if __name__=="__main__": main()
