from finder import Finder

if __name__ == "__main__":
    f = Finder()
    while True:
        print(*f.search(input(">>")), sep="\n")
