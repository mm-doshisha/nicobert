import sys


def main(argv):
    filename = argv[1]
    new_data = ""
    with open(filename, "r") as f:
        for line in f:
            if line.strip().startswith("#"):
                line = "\n"
            new_data += line

    with open(filename, "w") as f:
        f.write(new_data)


if __name__ == "__main__":
    main(sys.argv)
