from abc import ABC, abstractmethod

class Superclass(ABC):

    @abstractmethod
    def print_num(self, num):
        pass

    @abstractmethod
    def print_num_string(self, num):
        print('Printing string of %i' % num, '...')

class Subclass(Superclass):

    def print_num(self, num):
        print(num + 3)

    def print_num_string(self, num):
        super().print_num_string(num)
        for i in range(num):
            print(num)

n = Subclass()
n.print_num(7);
n.print_num_string(7);