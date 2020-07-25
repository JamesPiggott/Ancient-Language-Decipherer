from cuneiform.train_cuneiform import TrainCuneiform
from heiroglyphics.train_hieroglyphics import TrainHieroglyphics


class StartApplication:

    def __init__(self):
        while True:
            print()
            print("What do wish to analyze?")
            print("1. Cuneiform")
            print("2. Hieroglyphics")
            print("0. To exit the program")
            choice = input()
            if choice == '1':
                self.cuneiform()
            elif choice == '2':
                self.hieroglyphics()
            elif choice == '0':
                break
            else:
                print('Choice not recognized')

    def cuneiform(self):
        while True:
            print()
            print("Lets analyze Cuneiform")
            self.application_choice_menu()
            choice = input()
            if choice == '1':
                TrainCuneiform()
            elif choice == '2':
                print()
            elif choice == '0':
                break
            else:
                print('Choice not recognized')

    def hieroglyphics(self):
        while True:
            print()
            print("Lets analyze Hieroglyphics")
            self.application_choice_menu()
            choice = input()
            if choice == '1':
                hieroglyphics = TrainHieroglyphics()
                hieroglyphics.load_model()
            elif choice == '2':
                print()
            elif choice == '0':
                break
            else:
                print('Choice not recognized')

    def application_choice_menu(self):
        print("What do you want to do?")
        print("1. Train new model")
        print("2. Perform inference on model")
        print("0. To return to main menu?")

if __name__ == "__main__":

    """
    Start the application
    """
    app = StartApplication();