import subprocess


def main():
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    with open("commit_hash.txt", "w") as f:
        f.write(commit_hash)


if __name__ == "__main__":
    main()
