from train import train
from view_result import gen_picture
from config import SAVE_DIR

def main():
    history = train(1)
    gen_picture(history, SAVE_DIR)

if __name__ == '__main__':
    main()
